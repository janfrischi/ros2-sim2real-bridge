#!/usr/bin/env python3
"""
Behavior Cloning Policy Runner for Franka Robot
This node loads a trained BC policy using robomimic and runs inference on the Franka robot.
"""
import torch
import numpy as np
from typing import Dict, Optional, List
import argparse
import sys
import termios
import threading
import select
import json
import os
import random  # Add this import at the top
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from franka_msgs.action import Homing, Move, Grasp


# Import robomimic policy loading functionality
from robomimic.utils.file_utils import policy_from_checkpoint

# BCPolicy Runner Node
class BCPolicyRunner(Node):
    """ROS2 Node for running Behavior Cloning policy on Franka robot"""
    def __init__(self, policy_path: str, device: str = "cpu", deterministic: bool = True, 
                 control_frequency: float = 20.0, replay_file: str = None):
        super().__init__('bc_policy_runner')
        
        # Initialize parameters
        self.device = torch.device(device)
        self.deterministic = deterministic
        self.control_frequency = control_frequency
        self.step_count = 0
        
        # Replay mode parameters
        self.replay_mode = replay_file is not None
        self.replay_file = replay_file
        self.replay_data = None
        self.replay_index = 0
        self.replay_trial = 0
        self.replay_auto = False
        self.replay_execute_actions = False  # Toggle for executing actions in replay mode
        self.replay_auto_next_trial = False  # Auto-switch to next trial when current ends
        
        # Load policy using robomimic framework and set it to evaluation mode
        self.policy, self.ckpt_dict = self.load_policy(policy_path)

        # Initialize for new episode
        self.policy.start_episode()  
        
        # Robot state storage
        self.current_eef_pose = None
        self.current_gripper_positions = None

        # Object state storage -> TODO: Enable dynamic object state updates
        self.cube_positions = {
            'cube_1': np.array([0.4221598207950592, -0.1940348893404007, 0.0203000009059906]),
            'cube_2': np.array([0.47585567831993103, -0.046219781041145325, 0.0203000009059906]),
            'cube_3': np.array([0.4306733310222626, -0.2792506217956543, 0.0203000009059906])
        }
        
        # Cube orientations (quaternions) - w, x, y, z format "IsaacLab expects quaternions in [w, x, y, z] format"
        self.cube_quaternions = {
            'cube_1': np.array([0.0, 0.0, 0.0, 1.0]),  # Identity quaternion
            'cube_2': np.array([0.0, 0.0, 0.0, 1.0]),  # Identity quaternion
            'cube_3': np.array([0.0, 0.0, 0.0, 1.0])   # Identity quaternion
        }

        # Add cube spawning configuration
        self.cube_spawn_config = {
            "pose_range": {
                "x": (0.4, 0.6),
                "y": (-0.3, 0.3), 
                "z": (0.0203, 0.0203)
            },
            "min_cube_distance": 0.05  # Minimum distance between cubes to avoid overlap
        }

        # Environment origin (base frame reference)
        self.env_origin = np.array([0.0, 0.0, 0.0])  # Franka's home position in the world frame

        # Control flags
        self.is_running = False
        self.episode_active = False
        self.shutdown_requested = False
        
        # Keyboard input handler
        self.keyboard = KeyboardInput()
        
        # Callback group for allowing concurrent callbacks
        self.callback_group = ReentrantCallbackGroup()
        
        # --- Gripper Control Initialization ---
        self.gripper_goal_state = 'unknown' # 'open', 'closed', 'unknown'
        self.gripper_max_width = 0.08 # Max width for Franka Hand
        self.gripper_speed = 0.5 # Default speed (m/s)
        self.gripper_force = 50.0 # Default grasp force (N)
        self.gripper_epsilon_inner = 0.05
        self.gripper_epsilon_outer = 0.07
        
        # Grasping sequence and attachment tracking
        self.grasp_sequence_count = 0  # 0: no grasps, 1: first grasp (cube_2), 2: second grasp (cube_3)
        self.cube_attached = None      # None, 'cube_2', or 'cube_3'
        self.last_gripper_state = 'open'  # Track gripper state changes
        self.grasp_threshold = 0.02    # Width below which we consider gripper "closed"
        self.release_threshold = 0.06  # Width above which we consider gripper "open"
        self.gripper_state_stable_count = 0  # Counter for stable state detection
        self.required_stable_frames = 3      # Frames needed to confirm state change

        
        
        # Action clients for gripper, Homing, Move and Grasp are action definitions
        self.homing_client = ActionClient(self, Homing, '/fr3_gripper/homing', callback_group=self.callback_group)
        self.move_client = ActionClient(self, Move, '/fr3_gripper/move', callback_group=self.callback_group)
        self.grasp_client = ActionClient(self, Grasp, '/fr3_gripper/grasp', callback_group=self.callback_group)

        # Wait for gripper action servers
        self.wait_for_action_server(self.homing_client, 'Homing')
        self.wait_for_action_server(self.move_client, 'Move')
        self.wait_for_action_server(self.grasp_client, 'Grasp')
        # Perform initial homing
        self.home_gripper()
        # --- End Gripper Control Initialization ---
        
        # --- Enhanced Gripper Control Initialization (ADD THIS AFTER EXISTING GRIPPER INIT) ---
        #import threading
        from action_msgs.msg import GoalStatus

        # Enhanced gripper state management (minimal addition)
        self.gripper_action_in_progress = False
        self.gripper_action_lock = threading.Lock()
        self.gripper_last_command_time = 0.0
        self.gripper_command_cooldown = 1.0  # 1 second between commands
        # --- End Enhanced Gripper Control Initialization ---
        
        # Setup QoS (Quality of Service) profiles
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=1
        )
        
        # Load replay data if in replay mode
        if self.replay_mode:
            self.load_replay_data()
        
        # ------------------------------------------------------Subscribers--------------------------------------------------------------
        self.eef_pose_sub = self.create_subscription(
            PoseStamped,
            '/franka_robot_state_broadcaster/current_pose',
            self.eef_pose_callback,
            qos_profile,
            callback_group=self.callback_group
        )
        
        self.gripper_state_sub = self.create_subscription(
            JointState,
            '/fr3_gripper/joint_states',
            self.gripper_state_callback,
            qos_profile,
            callback_group=self.callback_group
        )
        
        # ------------------------------------------------------Publishers---------------------------------------------------------------
        self.pose_command_pub = self.create_publisher(
            Float64MultiArray,
            '/cartesian_position_controller/commands',
            qos_profile
        )

        self.gripper_command_pub = self.create_publisher(
            Float64MultiArray,
            '/gripper_position_controller/commands',
            qos_profile
        )

        # Debug observation publisher
        self.observation_debug_pub = self.create_publisher(
            Float64MultiArray,
            '/bc_policy/observation_debug',
            qos_profile
        )
        
        # --------------------------------- Single timer for both normal and replay modes ---------------------------------
        self.control_timer = self.create_timer(
            1.0 / self.control_frequency,  # Period = 1/20Hz = 0.05 seconds
            self.unified_control_loop,  # Single unified method for both modes
            callback_group=self.callback_group,
            clock=rclpy.clock.Clock(clock_type=rclpy.clock.ClockType.STEADY_TIME)
        )
        
        # Keyboard input timer (check for keypress every 50ms)
        self.keyboard_timer = self.create_timer(0.05, self.check_keyboard_input)
        
        # Print initial instructions
        self.print_instructions()

    def load_policy(self, policy_path: str):
        """Load the trained BC policy using robomimic's policy_from_checkpoint"""
        try:
            self.get_logger().info(f"Loading policy from: {policy_path}")
            
            # Use robomimic's policy_from_checkpoint function "Abstracts away the loading logic"
            policy, ckpt_dict = policy_from_checkpoint(
                device=self.device,
                ckpt_path=policy_path,
                verbose=True
            )
            
            self.get_logger().info("Successfully loaded policy using robomimic")
            self.get_logger().info(f"Algorithm: {ckpt_dict.get('algo_name', 'Unknown')}")
            
            # Log shape metadata if available
            if 'shape_metadata' in ckpt_dict:
                shape_meta = ckpt_dict['shape_metadata']
                self.get_logger().info(f"Action dimension: {shape_meta.get('ac_dim', 'Unknown')}")
                self.get_logger().info(f"Observation keys: {list(shape_meta.get('all_shapes', {}).keys())}")
            
            return policy, ckpt_dict
            
        except Exception as e:
            self.get_logger().error(f"Error loading policy: {e}")
            raise

    # Load replay data from JSON file "Data recorded during inference in IsaacSim"
    def load_replay_data(self):
        """Load observation data from JSON file for replay mode"""
        try:
            if not os.path.exists(self.replay_file):
                raise FileNotFoundError(f"Replay file not found: {self.replay_file}")
                
            with open(self.replay_file, 'r') as f:
                self.replay_data = json.load(f)
                
            print(f"✅ Loaded replay data with {len(self.replay_data)} trials")
            for i, trial in enumerate(self.replay_data):
                num_obs = len(trial['observations'])
                total_steps = trial['metadata']['total_steps']
                print(f"   Trial {i}: {num_obs} observations, {total_steps} total steps")
                
        except Exception as e:
            self.get_logger().error(f"Failed to load replay data: {e}")
            raise

    # Convert flat observation array back to policy format
    def parse_replay_observation(self, obs_data: Dict) -> Dict[str, np.ndarray]:
        """Convert observation data from JSON to policy input format"""
        try:
            # Extract the arrays - these should be single values since they're from the JSON
            eef_pos = np.array(obs_data['eef_pos'], dtype=np.float32)      # [3]
            eef_quat = np.array(obs_data['eef_quat'], dtype=np.float32)    # [4] 
            gripper_pos = np.array(obs_data['gripper_pos'], dtype=np.float32)  # [2]
            object_obs = np.array(obs_data['object'], dtype=np.float32)    # [39]
            
            # Return as numpy arrays for robomimic policy
            obs_dict = {
                'eef_pos': eef_pos,
                'eef_quat': eef_quat,
                'gripper_pos': gripper_pos,
                'object': object_obs
            }
            
            return obs_dict
            
        except Exception as e:
            self.get_logger().error(f"Error parsing replay observation: {e}")
            import traceback
            traceback.print_exc()
            return None

    # Step through replay data - run policy inference on the observations   
    def replay_step(self):
        """Step to next observation in replay mode"""
        if not self.replay_mode or not self.replay_data:
            return
            
        current_trial = self.replay_data[self.replay_trial]
        observations = current_trial['observations']
        
        if self.replay_index >= len(observations):
            print(f"\n🏁 End of trial {self.replay_trial + 1} reached")
            
            # Auto-switch to next trial if enabled and more trials available
            if self.replay_auto_next_trial and self.replay_trial + 1 < len(self.replay_data):
                print(f"🔄 Auto-switching to trial {self.replay_trial + 2}")
                self.replay_next_trial()
                return
            elif self.replay_trial + 1 >= len(self.replay_data):
                print(f"🏁 All trials completed! ({len(self.replay_data)} trials)")
                self.replay_auto = False  # Stop auto-stepping
                return
            else:
                # Auto-switch disabled, just stop auto-stepping
                self.replay_auto = False
                return
            
        # Get current observation from JSON
        obs_data = observations[self.replay_index]
        
        # Parse observation
        obs_dict = self.parse_replay_observation(obs_data)
            
        # Run policy inference using robomimic policy
        try:
            action = self.policy(obs_dict)
            action_np = action if isinstance(action, np.ndarray) else action.cpu().numpy()
            
            # # Log observation and action together (same as normal mode)
            # if obs_dict is not None:
            #     self.log_observation_compact(obs_dict)
            
            # Execute the action only if action execution is enabled 
            if self.replay_execute_actions:
                self.execute_action(action_np)
            
        except Exception as e:
            self.get_logger().error(f"Policy inference failed: {e}")
            return
            
        # Get recorded action if available
        recorded_action = obs_data.get('action', [])
        
        # Display comparison
        self.display_replay_comparison(obs_data, action_np, recorded_action)
        
        # Advance to next observation
        self.replay_index += 1

    def detect_gripper_state_change(self) -> Optional[str]:
        """
        Detect reliable gripper state changes (open->closed or closed->open)
        Returns 'closed', 'open', or None if no reliable change detected
        """
        if self.current_gripper_positions is None:
            return None
        
        # Calculate current gripper width
        current_width = abs(self.current_gripper_positions[0]) + abs(self.current_gripper_positions[1])
        
        # Determine current state based on width
        if current_width < self.grasp_threshold:
            current_state = 'closed'
        elif current_width > self.release_threshold:
            current_state = 'open'
        else:
            # In between - maintain previous state
            current_state = self.last_gripper_state
        
        # Check if state has changed
        if current_state != self.last_gripper_state:
            self.gripper_state_stable_count += 1
            
            # Only register change after stable frames
            if self.gripper_state_stable_count >= self.required_stable_frames:
                previous_state = self.last_gripper_state
                self.last_gripper_state = current_state
                self.gripper_state_stable_count = 0
                
                self.get_logger().info(f"Gripper state change detected: {previous_state} -> {current_state}")
                return current_state
        else:
            # State is stable, reset counter
            self.gripper_state_stable_count = 0
        
        return None
    
    def handle_cube_attachment(self, gripper_state_change: str):
        """
        Handle cube attachment/detachment based on gripper state changes
        """
        if gripper_state_change == 'closed':
            # Gripper just closed - attach next cube in sequence
            if self.grasp_sequence_count == 0:
                # First grasp - attach cube_2
                self.cube_attached = 'cube_2'
                self.grasp_sequence_count = 1
                print(f"🔴 CUBE_2 (Red) ATTACHED to gripper - Dynamic tracking enabled")
                self.get_logger().info("Cube_2 attached - enabling dynamic position tracking")
                
            elif self.grasp_sequence_count == 1 and self.cube_attached is None:
                # Second grasp - attach cube_3
                self.cube_attached = 'cube_3'
                self.grasp_sequence_count = 2
                print(f"🟢 CUBE_3 (Green) ATTACHED to gripper - Dynamic tracking enabled")
                self.get_logger().info("Cube_3 attached - enabling dynamic position tracking")
                
            else:
                print(f"⚠️ Unexpected grasp detected - ignoring")
        
        elif gripper_state_change == 'open':
            # Gripper just opened - detach current cube
            if self.cube_attached == 'cube_2':
                print(f"🔴 CUBE_2 (Red) RELEASED - Dynamic tracking disabled")
                self.get_logger().info("Cube_2 released - disabling dynamic position tracking")
                self.cube_attached = None
                
            elif self.cube_attached == 'cube_3':
                print(f"🟢 CUBE_3 (Green) RELEASED - Dynamic tracking disabled")
                self.get_logger().info("Cube_3 released - disabling dynamic position tracking")
                self.cube_attached = None
                
            else:
                print(f"⚠️ Gripper opened but no cube was attached")


    def update_attached_cube_pose(self):
        """
        Update the position and orientation of the attached cube to match the end-effector
        """
        if self.cube_attached is None or self.current_eef_pose is None:
            return
        
        # Get current end-effector pose
        eef_pos = np.array([
            self.current_eef_pose.pose.position.x,
            self.current_eef_pose.pose.position.y,
            self.current_eef_pose.pose.position.z
        ])
        
        # Get current end-effector quaternion [qx, qy, qz, qw] from ROS
        eef_quat_ros = np.array([
            self.current_eef_pose.pose.orientation.x,  # qx
            self.current_eef_pose.pose.orientation.y,  # qy
            self.current_eef_pose.pose.orientation.z,  # qz
            self.current_eef_pose.pose.orientation.w   # qw
        ])
        
        # Convert to [qw, qx, qy, qz] format for cube storage (IsaacLab format)
        eef_quat_isaac = np.array([
            eef_quat_ros[3],  # qw
            eef_quat_ros[0],  # qx
            eef_quat_ros[1],  # qy
            eef_quat_ros[2]   # qz
        ])
        
        # Apply slight offset to cube position (cube is slightly below gripper)
        cube_offset = np.array([0.0, 0.0, 0.0])  # 5cm below gripper
        attached_cube_pos = eef_pos + cube_offset
        
        # Update the attached cube's pose
        self.cube_positions[self.cube_attached] = attached_cube_pos
        self.cube_quaternions[self.cube_attached] = eef_quat_isaac
        
        # Optional: Log the update (comment out if too verbose)
        # self.get_logger().debug(f"Updated {self.cube_attached} pose: pos={attached_cube_pos}, quat={eef_quat_isaac}")

    def display_replay_comparison(self, obs_data: Dict, policy_action: np.ndarray, recorded_action: List):
        """Display comparison between recorded observation and policy output"""
        step = obs_data['step']
        timestamp = obs_data['timestamp']

        # Force terminal synchronization
        sys.stdout.flush()
        sys.stderr.flush()
        
        # Use explicit newlines and avoid centering/formatting that might be affected by terminal mode
        print("\n" + "="*80)
        print(f"REPLAY STEP {step} (t={timestamp:.2f}s) - Trial {self.replay_trial}")
        print("="*80)
        print()

        # Show observation breakdown
        print("📥 INPUT OBSERVATION:")
        eef_pos = obs_data['eef_pos']
        eef_quat = obs_data['eef_quat']
        gripper_pos = obs_data['gripper_pos']
        print(f"   EEF Position:       [{eef_pos[0]:+7.4f}, {eef_pos[1]:+7.4f}, {eef_pos[2]:+7.4f}]")
        print(f"   EEF Quaternion:     [{eef_quat[0]:+7.4f}, {eef_quat[1]:+7.4f}, {eef_quat[2]:+7.4f}, {eef_quat[3]:+7.4f}]")
        print(f"   Gripper Pos:        [{gripper_pos[0]:+7.4f}, {gripper_pos[1]:+7.4f}]")
        print(f"   Object state -------------------------------------------------------")
        print(f"   Cube 1 Positions:   [{obs_data['object'][0]:+7.4f}, {obs_data['object'][1]:+7.4f}, {obs_data['object'][2]:+7.4f}]")
        print(f"   Cube 1 Quaternion:  [{obs_data['object'][3]:+7.4f}, {obs_data['object'][4]:+7.4f}, {obs_data['object'][5]:+7.4f}, {obs_data['object'][6]:+7.4f}]")
        print(f"   Cube 2 Positions:   [{obs_data['object'][7]:+7.4f}, {obs_data['object'][8]:+7.4f}, {obs_data['object'][9]:+7.4f}]")
        print(f"   Cube 2 Quaternion:  [{obs_data['object'][10]:+7.4f}, {obs_data['object'][11]:+7.4f}, {obs_data['object'][12]:+7.4f}, {obs_data['object'][13]:+7.4f}]")
        print(f"   Cube 3 Positions:   [{obs_data['object'][14]:+7.4f}, {obs_data['object'][15]:+7.4f}, {obs_data['object'][16]:+7.4f}]")
        print(f"   Cube 3 Quaternion:  [{obs_data['object'][17]:+7.4f}, {obs_data['object'][18]:+7.4f}, {obs_data['object'][19]:+7.4f}, {obs_data['object'][20]:+7.4f}]")
        print(f"   Gripper to Cube 1:  [{obs_data['object'][21]:+7.4f}, {obs_data['object'][22]:+7.4f}, {obs_data['object'][23]:+7.4f}]")
        print(f"   Gripper to Cube 2:  [{obs_data['object'][24]:+7.4f}, {obs_data['object'][25]:+7.4f}, {obs_data['object'][26]:+7.4f}]")
        print(f"   Gripper to Cube 3:  [{obs_data['object'][27]:+7.4f}, {obs_data['object'][28]:+7.4f}, {obs_data['object'][29]:+7.4f}]")
        print(f"   Cube 1 to Cube 2: [{obs_data['object'][30]:+7.4f}, {obs_data['object'][31]:+7.4f}, {obs_data['object'][32]:+7.4f}]")
        print(f"   Cube 2 to Cube 3: [{obs_data['object'][33]:+7.4f}, {obs_data['object'][34]:+7.4f}, {obs_data['object'][35]:+7.4f}]")
        print(f"   Cube 1 to Cube 3: [{obs_data['object'][36]:+7.4f}, {obs_data['object'][37]:+7.4f}, {obs_data['object'][38]:+7.4f}]")
        print()

        print("🎬 POLICY OUTPUT:")
        print(f"   Full Action:     [{', '.join([f'{x:+7.4f}' for x in policy_action])}]")
        print(f"   EEF Pose:        [{', '.join([f'{x:+7.4f}' for x in policy_action[:7]])}] (x,y,z,qw,qx,qy,qz)")
        print(f"   Gripper Cmd:     {policy_action[7]:+7.4f}")
        print()

        if recorded_action:
            print("📹 RECORDED ACTION:")
            print(f"   Full Action:     [{', '.join([f'{x:+7.4f}' for x in recorded_action])}]")
            print(f"   EEF Pose:        [{', '.join([f'{x:+7.4f}' for x in recorded_action[:7]])}]")
            print(f"   Gripper Cmd:     {recorded_action[7]:+7.4f}")
            print()

            # Calculate differences
            if len(recorded_action) == len(policy_action):
                diff = np.abs(np.array(policy_action) - np.array(recorded_action))
                print("📊 DIFFERENCES:")
                print(f"   Abs Difference:  [{', '.join([f'{x:+7.4f}' for x in diff])}]")
                print(f"   Max Difference:  {np.max(diff):+7.4f}")
                print(f"   Mean Difference: {np.mean(diff):+7.4f}")
                print()

        print("="*80)
        print()
        
        # Force flush after all output
        sys.stdout.flush()
        sys.stderr.flush()

    # Auto-step through replay data
    def replay_auto_step(self):
        """Simplified replay auto-step - timing handled by unified ROS2 timer"""
        # This method is now simplified since timing is handled by the unified timer
        # No manual timing checks needed - just execute the step
        if self.replay_auto and self.replay_mode:
            self.replay_step()

    # Reset replay to beginning
    def replay_reset(self):
        """Reset replay to beginning of current trial"""
        if not self.replay_mode:
            return
            
        self.replay_index = 0
        self.policy.start_episode()  # Reset policy state for new episode
        print(f"🔄 Reset replay to beginning of trial {self.replay_trial}")

    # Switch to different trial
    def replay_next_trial(self):
        """Switch to next trial in replay data"""
        if not self.replay_mode or not self.replay_data:
            return
            
        if self.replay_trial < len(self.replay_data) - 1:
            self.replay_trial += 1
            self.replay_reset()
            print(f"➡️ Switched to trial {self.replay_trial}")
        else:
            print("Already at last trial")

    def replay_prev_trial(self):
        """Switch to previous trial in replay data"""
        if not self.replay_mode or not self.replay_data:
            return
            
        if self.replay_trial > 0:
            self.replay_trial -= 1
            self.replay_reset()
            print(f"⬅️ Switched to trial {self.replay_trial}")
        else:
            print("Already at first trial")

    # Unified action execution method for both normal and replay modes
    def execute_action(self, action_np: np.ndarray):
        """Unified action execution for both normal and replay modes"""
        try:
            # Ensure it's a 1D array
            if action_np.ndim > 1:
                action_np = action_np.squeeze()
            
            # Interpret action - 7D end-effector pose + 1D gripper
            eef_pose = action_np[:7]  # [x, y, z, qw, qx, qy, qz] - IsaacLab format
            gripper_command = action_np[7]  # Gripper command
                
            # Extract position and quaternion from pose
            position = eef_pose[:3]  # [x, y, z]
            quaternion_sim = eef_pose[3:]  # [qw, qx, qy, qz] - IsaacLab format
            
            # Convert from IsaacLab [qw, qx, qy, qz] to ROS [qx, qy, qz, qw]
            quaternion_ros = np.array([
                quaternion_sim[1],  # qx
                quaternion_sim[2],  # qy
                quaternion_sim[3],  # qz
                quaternion_sim[0]   # qw
            ])
            
            # Normalize quaternion to ensure it's valid
            quat_norm = np.linalg.norm(quaternion_ros)
            if quat_norm > 0:
                quaternion_ros = quaternion_ros / quat_norm
            else:
                self.get_logger().warn("Invalid quaternion, skipping action execution")
                return
            
            # --- UNIFIED Gripper Control Logic ---
            # Clamp gripper command to expected range [-1, 1]
            gripper_command = np.clip(gripper_command, -1.0, 1.0)
            
            # Determine desired gripper state based on command
            desired_gripper_state = 'closed' if gripper_command < 0 else 'open'
            if desired_gripper_state != self.gripper_goal_state:
                if desired_gripper_state == 'open':
                    self.open_gripper()
                elif desired_gripper_state == 'closed':
                    self.close_gripper()
            
            # Create cartesian pose command: [x, y, z, qx, qy, qz, qw]
            cartesian_pose = np.concatenate([
                position,         # [x, y, z]
                quaternion_ros    # [qx, qy, qz, qw]
            ])
            
            # Publish cartesian pose commands to the controller
            pose_msg = Float64MultiArray()
            pose_msg.data = cartesian_pose.tolist()
            self.pose_command_pub.publish(pose_msg)

            self.get_logger().debug(f"Action executed: pos={position}, quat={quaternion_ros}, gripper={gripper_command}")
            
        except Exception as e:
            self.get_logger().error(f"Error executing action: {e}")

    # Helper to wait for action servers
    def wait_for_action_server(self, client, name):
        self.get_logger().info(f'Waiting for {name} action server...')
        while not client.wait_for_server(timeout_sec=2.0) and rclpy.ok():
            self.get_logger().info(f'{name} action server not available, waiting again...')
        if rclpy.ok():
            self.get_logger().info(f'{name} action server found.')
        else:
             self.get_logger().error(f'ROS shutdown while waiting for {name} server.')
             raise SystemExit('ROS shutdown')

    # Define gripper control methods
    def home_gripper(self):
        goal_msg = Homing.Goal()
        # Send goal async and forget (or handle future if needed)
        self.homing_client.send_goal_async(goal_msg)
        self.gripper_goal_state = 'open' # Assume homing opens the gripper

    def open_gripper(self):
        """Open the gripper using the action client with safety checks"""
        with self.gripper_action_lock:
            current_time = time.time()
            
            # Safety checks
            if self.gripper_action_in_progress:
                self.get_logger().debug("Gripper action in progress, skipping open command")
                return
            
            if current_time - self.gripper_last_command_time < self.gripper_command_cooldown:
                self.get_logger().debug("Gripper cooldown active, skipping open command")
                return
                
            if self.gripper_goal_state == 'open':
                return  # Already open
                
            # Set state and send command
            self.gripper_action_in_progress = True
            self.gripper_last_command_time = current_time
        
        try:
            goal_msg = Move.Goal()
            goal_msg.width = self.gripper_max_width
            goal_msg.speed = self.gripper_speed
            
            # Send goal with result callback
            goal_future = self.move_client.send_goal_async(goal_msg)
            goal_future.add_done_callback(self._gripper_goal_callback)
            
            self.gripper_goal_state = 'open'
            self.get_logger().debug("Safe gripper OPEN command sent")
            
        except Exception as e:
            with self.gripper_action_lock:
                self.gripper_action_in_progress = False
            self.get_logger().error(f"Failed to send gripper open command: {e}")

    def close_gripper(self):
        """Close the gripper using the action client with safety checks"""
        with self.gripper_action_lock:
            current_time = time.time()
            
            # Safety checks
            if self.gripper_action_in_progress:
                self.get_logger().debug("Gripper action in progress, skipping close command")
                return
            
            if current_time - self.gripper_last_command_time < self.gripper_command_cooldown:
                self.get_logger().debug("Gripper cooldown active, skipping close command")
                return
                
            if self.gripper_goal_state == 'closed':
                return  # Already closed
                
            # Set state and send command
            self.gripper_action_in_progress = True
            self.gripper_last_command_time = current_time
        
        try:
            goal_msg = Grasp.Goal()
            goal_msg.width = 0.0
            goal_msg.speed = self.gripper_speed
            goal_msg.force = self.gripper_force
            goal_msg.epsilon.inner = self.gripper_epsilon_inner
            goal_msg.epsilon.outer = self.gripper_epsilon_outer

            # Send goal with result callback
            goal_future = self.grasp_client.send_goal_async(goal_msg)
            goal_future.add_done_callback(self._gripper_goal_callback)
            
            self.gripper_goal_state = 'closed'
            self.get_logger().debug("Safe gripper CLOSE command sent")
            
        except Exception as e:
            with self.gripper_action_lock:
                self.gripper_action_in_progress = False
            self.get_logger().error(f"Failed to send gripper close command: {e}")

    def _gripper_goal_callback(self, future):
        """Minimal callback to reset gripper action state"""
        try:
            goal_handle = future.result()
            if goal_handle.accepted:
                # Get result to reset state when complete
                result_future = goal_handle.get_result_async()
                result_future.add_done_callback(self._gripper_result_callback)
            else:
                with self.gripper_action_lock:
                    self.gripper_action_in_progress = False
        except Exception:
            with self.gripper_action_lock:
                self.gripper_action_in_progress = False

    def _gripper_result_callback(self, future):
        """Reset gripper state when action completes"""
        with self.gripper_action_lock:
            self.gripper_action_in_progress = False

    def eef_pose_callback(self, msg: PoseStamped):
        """Callback for end-effector pose updates"""
        self.current_eef_pose = msg # Quaternion is in x, y, z, w format
    
    def gripper_state_callback(self, msg: JointState):
        """Callback for gripper state updates"""
        # Gripper should have symmetric but opposite values: [+value, -value] "IsaacLab Convention"
        finger_1_pos = msg.position[0]  # First finger (positive)
        finger_2_pos = -msg.position[1] # Second finger (negative)
        self.current_gripper_positions = np.array([finger_1_pos, finger_2_pos])

    def randomly_spawn_cubes(self):
        """Randomly spawn the three cubes within the specified range"""
        print("\n🎲 RANDOMLY SPAWNING CUBES")
        print("=" * 50)
        
        pose_range = self.cube_spawn_config["pose_range"]
        min_distance = self.cube_spawn_config["min_cube_distance"]
        
        new_positions = {}
        cube_names = ['cube_1', 'cube_2', 'cube_3']
        
        for i, cube_name in enumerate(cube_names):
            max_attempts = 50  # Prevent infinite loop
            attempts = 0
            
            while attempts < max_attempts:
                # Generate random position
                x = random.uniform(pose_range["x"][0], pose_range["x"][1])
                y = random.uniform(pose_range["y"][0], pose_range["y"][1])
                z = pose_range["z"][0]  # Fixed Z value (table height)
                
                new_pos = np.array([x, y, z])
                
                # Check distance from existing cubes
                valid_position = True
                for existing_cube, existing_pos in new_positions.items():
                    distance = np.linalg.norm(new_pos - existing_pos)
                    if distance < min_distance:
                        valid_position = False
                        break
                
                if valid_position:
                    new_positions[cube_name] = new_pos
                    break
                
                attempts += 1
            
            if attempts >= max_attempts:
                # Fallback: use a safe position if we can't find a valid random one
                fallback_positions = {
                    'cube_1': np.array([0.45, -0.15, 0.0203]),
                    'cube_2': np.array([0.5, 0.0, 0.0203]),
                    'cube_3': np.array([0.55, 0.15, 0.0203])
                }
                new_positions[cube_name] = fallback_positions[cube_name]
                print(f"⚠️ Using fallback position for {cube_name} after {max_attempts} attempts")
        
    
        self.cube_positions.update(new_positions)
        
        # Display the changes
        print("📍 NEW CUBE POSITIONS:")
        color_names = {
            'cube_1': 'Blue Cube  ',
            'cube_2': 'Red Cube   ',
            'cube_3': 'Green Cube '
        }
        for cube_name, pos in new_positions.items():
            print(f"  {color_names[cube_name]}: [{pos[0]:+7.4f}, {pos[1]:+7.4f}, {pos[2]:+7.4f}]")
            
        
        print("=" * 50)
        print("✅ Cube spawning completed!")
        
        # Reset policy episode state since environment changed
        if hasattr(self, 'policy') and self.policy:
            self.policy.start_episode()
            print("🔄 Policy episode state reset due to environment change")

    def spawn_cubes_preset(self, preset_name: str = "default"):
        """Spawn cubes using predefined pose presets"""
        print(f"\n🎯 SPAWNING CUBES - PRESET: {preset_name.upper()}")
        print("=" * 50)
        
        # Define all preset configurations
        presets = {
            "default": {
                'cube_1': np.array([0.4221598207950592, -0.1940348893404007, 0.0203]),
                'cube_2': np.array([0.47585567831993103, -0.046219781041145325, 0.0203]),
                'cube_3': np.array([0.4306733310222626, -0.2792506217956543, 0.0203])
            },
            "custom_1": {
                'cube_1': np.array([0.400, 0.200, 0.020]),
                'cube_2': np.array([0.600, 0.300, 0.020]),
                'cube_3': np.array([0.400, -0.200, 0.020])
            },
            "wide_spread": {
                'cube_1': np.array([0.35, -0.25, 0.0203]),
                'cube_2': np.array([0.65, 0.0, 0.0203]),
                'cube_3': np.array([0.45, 0.25, 0.0203])
            },
            "tight_cluster": {
                'cube_1': np.array([0.50, -0.08, 0.0203]),
                'cube_2': np.array([0.50, 0.0, 0.0203]),
                'cube_3': np.array([0.50, 0.08, 0.0203])
            },
            "corner_formation": {
                'cube_1': np.array([0.40, -0.20, 0.0203]),  # Bottom left
                'cube_2': np.array([0.40, 0.20, 0.0203]),   # Top left
                'cube_3': np.array([0.60, 0.0, 0.0203])     # Right center
            },
            "stacking_ready": {
                'cube_1': np.array([0.50, 0.0, 0.0203]),    # Target base
                'cube_2': np.array([0.40, -0.15, 0.0203]),  # Source 1
                'cube_3': np.array([0.60, 0.15, 0.0203])    # Source 2
            },
            "manipulation_test": {
                'cube_1': np.array([0.45, -0.10, 0.0203]),
                'cube_2': np.array([0.55, 0.10, 0.0203]),
                'cube_3': np.array([0.50, 0.0, 0.0203])
            },
            "reach_challenge": {
                'cube_1': np.array([0.35, -0.30, 0.0203]),  # Far left
                'cube_2': np.array([0.65, 0.30, 0.0203]),   # Far right
                'cube_3': np.array([0.50, 0.0, 0.0203])     # Center
            },
            "pick_place_demo": {
                'cube_1': np.array([0.42, -0.18, 0.0203]),  # Pick source
                'cube_2': np.array([0.58, 0.18, 0.0203]),   # Place target area
                'cube_3': np.array([0.50, -0.05, 0.0203])   # Obstacle/intermediate
            },
            "sorting_task": {
                'cube_1': np.array([0.38, -0.25, 0.0203]),  # Left bin
                'cube_2': np.array([0.50, 0.0, 0.0203]),    # Center (to sort)
                'cube_3': np.array([0.62, 0.25, 0.0203])    # Right bin
            },
            "assembly_line": {
                'cube_1': np.array([0.40, 0.0, 0.0203]),    # Input
                'cube_2': np.array([0.50, 0.0, 0.0203]),    # Processing
                'cube_3': np.array([0.60, 0.0, 0.0203])     # Output
            },
            "circular_arrangement": {
                'cube_1': np.array([0.50, -0.12, 0.0203]),  # Bottom
                'cube_2': np.array([0.44, 0.06, 0.0203]),   # Top left
                'cube_3': np.array([0.56, 0.06, 0.0203])    # Top right
            },
            "precision_test": {
                'cube_1': np.array([0.48, -0.05, 0.0203]),
                'cube_2': np.array([0.50, 0.0, 0.0203]),
                'cube_3': np.array([0.52, 0.05, 0.0203])
            },
            "learning_progression_1": {
                'cube_1': np.array([0.45, -0.15, 0.0203]),  # Easy reach
                'cube_2': np.array([0.50, 0.0, 0.0203]),    # Medium
                'cube_3': np.array([0.55, 0.15, 0.0203])    # Harder reach
            },
            "learning_progression_2": {
                'cube_1': np.array([0.40, -0.20, 0.0203]),  # Further challenge
                'cube_2': np.array([0.60, 0.20, 0.0203]),   # Cross workspace
                'cube_3': np.array([0.50, 0.0, 0.0203])     # Central reference
            },
            "workspace_corners": {
                'cube_1': np.array([0.35, -0.30, 0.0203]),  # Bottom left corner
                'cube_2': np.array([0.35, 0.30, 0.0203]),   # Top left corner
                'cube_3': np.array([0.65, 0.0, 0.0203])     # Right edge
            }
        }
    
        # Check if preset exists
        if preset_name not in presets:
            available_presets = list(presets.keys())
            print(f"❌ Unknown preset: {preset_name}")
            print(f"📋 Available presets: {', '.join(available_presets)}")
            return
        
        # Get the preset positions
        positions = presets[preset_name]
        
        # Update cube positions
        self.cube_positions.update(positions)
        
        # Reset orientations to identity for all presets
        for cube_name in ['cube_1', 'cube_2', 'cube_3']:
            self.cube_quaternions[cube_name] = np.array([0.0, 0.0, 0.0, 1.0])
        
        # Display new positions with enhanced formatting
        color_names = {
            'cube_1': '🔵 Blue Cube ',
            'cube_2': '🔴 Red Cube  ',
            'cube_3': '🟢 Green Cube'
        }
        
        print("📍 NEW CUBE POSITIONS:")
        for cube_name, pos in positions.items():
            print(f"  {color_names[cube_name]}: [{pos[0]:+7.4f}, {pos[1]:+7.4f}, {pos[2]:+7.4f}]")
        
        # Calculate workspace metrics
        distances = []
        cube_positions_list = list(positions.values())
        for i in range(len(cube_positions_list)):
            for j in range(i+1, len(cube_positions_list)):
                dist = np.linalg.norm(cube_positions_list[i] - cube_positions_list[j])
                distances.append(dist)
        
        min_distance = min(distances)
        max_distance = max(distances)
        avg_distance = np.mean(distances)
        
        print(f"\n📊 WORKSPACE METRICS:")
        print(f"   Min distance between cubes: {min_distance:.4f}m")
        print(f"   Max distance between cubes: {max_distance:.4f}m")
        print(f"   Avg distance between cubes: {avg_distance:.4f}m")
        
        print(f"\n✅ Preset '{preset_name}' applied successfully!")
        print("=" * 50)
        
        # Reset policy episode state
        if hasattr(self, 'policy') and self.policy:
            self.policy.start_episode()
            print("🔄 Policy episode state reset due to environment change")

    def list_cube_presets(self):
        """List all available cube presets with descriptions"""
        presets_info = {
            "default": "Original IsaacLab training positions",
            "custom_1": "Your requested custom positions",
            "wide_spread": "Cubes spread across full workspace",
            "tight_cluster": "Cubes close together in center",
            "corner_formation": "L-shaped corner arrangement",
            "stacking_ready": "Optimal positions for stacking tasks",
            "manipulation_test": "Standard manipulation testing layout",
            "reach_challenge": "Tests maximum reach capabilities",
            "pick_place_demo": "Demonstration of pick-and-place",
            "sorting_task": "Three-bin sorting scenario",
            "assembly_line": "Linear assembly sequence",
            "circular_arrangement": "Triangular/circular formation",
            "precision_test": "Close spacing for precision testing",
            "learning_progression_1": "Beginner difficulty progression",
            "learning_progression_2": "Advanced difficulty progression",
            "workspace_corners": "Extreme workspace positions"
        }
        
        print("\n📋 AVAILABLE CUBE PRESETS")
        print("=" * 60)
        for preset, description in presets_info.items():
            print(f"  {preset:<22} │ {description}")
        print("=" * 60)
        print("Usage: Press the corresponding number key or use 'p' + preset name")
        print()

    def spawn_cubes_in_pattern(self, pattern: str = "line"):
        """Spawn cubes in predefined patterns"""
        print(f"\n📐 SPAWNING CUBES IN {pattern.upper()} PATTERN")
        print("=" * 50)
        
        if pattern == "line":
            # Cubes in a line from left to right
            positions = {
                'cube_1': np.array([0.5, -0.2, 0.0203]),
                'cube_2': np.array([0.5, 0.0, 0.0203]),
                'cube_3': np.array([0.5, 0.2, 0.0203])
            }
        elif pattern == "triangle":
            # Cubes in a triangle formation
            positions = {
                'cube_1': np.array([0.45, -0.1, 0.0203]),
                'cube_2': np.array([0.45, 0.1, 0.0203]),
                'cube_3': np.array([0.55, 0.0, 0.0203])
            }
        elif pattern == "stack_ready":
            # Cubes positioned for easy stacking
            positions = {
                'cube_1': np.array([0.5, 0.0, 0.0203]),      # Bottom (target)
                'cube_2': np.array([0.45, -0.15, 0.0203]),   # Source 1
                'cube_3': np.array([0.55, 0.15, 0.0203])     # Source 2
            }
        else:
            print(f"❌ Unknown pattern: {pattern}")
            return
        
        # Update positions
        self.cube_positions.update(positions)
        
        # Reset orientations to identity
        for cube_name in ['cube_1', 'cube_2', 'cube_3']:
            self.cube_quaternions[cube_name] = np.array([0.0, 0.0, 0.0, 1.0])
        
        # Display new positions
        color_names = {
            'cube_1': 'Blue Cube  ',
            'cube_2': 'Red Cube   ',
            'cube_3': 'Green Cube '
        }
        
        print("📍 NEW CUBE POSITIONS:")
        for cube_name, pos in positions.items():
            print(f"  {color_names[cube_name]}: [{pos[0]:+7.4f}, {pos[1]:+7.4f}, {pos[2]:+7.4f}]")
        
        print(f"✅ {pattern.capitalize()} pattern applied!")
        print("=" * 50)
        
        # Reset policy episode state
        if hasattr(self, 'policy') and self.policy:
            self.policy.start_episode()
            print("🔄 Policy episode state reset due to environment change")

    def print_instructions(self):
        """Print control instructions"""
        print("\n" + "=" * 90)
        print("BC POLICY RUNNER - CONTROL INSTRUCTIONS".center(90))
        print("=" * 90)
        
        if self.replay_mode and self.replay_data:
            # ... existing replay mode instructions ...
            pass
        
        print("Robot Controls:")
        print("  Space-Bar: Start policy execution")
        print("  s: Stop policy execution")
        print("  r: Reset to home position")
        print("  o: Toggle gripper (open/close)")
        print("-" * 90)
        print("Environment Controls - Basic Patterns:")
        print("  c: Random spawn      │ 1: Line pattern      │ 2: Triangle pattern  │ 3: Stack-ready")
        print("-" * 90)
        print("Environment Controls - Number Key Presets:")
        print("  4: Default           │ 5: Custom positions  │ 6: Wide spread       │ 7: Tight cluster")
        print("  8: Corner formation  │ 9: Stacking ready    │ 0: Manipulation test")
        print("-" * 90)
        print("Environment Controls - Letter Key Presets:")
        print("  a: Reach challenge   │ b: Pick-place demo   │ d: Sorting task       │ e: Assembly line")
        print("  f: Circular arrange  │ h: Precision test    │ i: Learning prog 1    │ j: Learning prog 2")
        print("  k: Workspace corners")
        print("-" * 90)
        print("Information & Control:")
        print("  l: List all presets  │ g: Emergency gripper reset │ q: Quit")
        print("=" * 90)
        print("💡 TIP: Use 'l' to see detailed descriptions of all presets")
        print()

    def compute_object_observations(self) -> np.ndarray:
        """
        Compute 39D object observations matching IsaacLab structure:
        - cube_1 pos (3D) - relative to env origin
        - cube_1 quat (4D) 
        - cube_2 pos (3D) - relative to env origin
        - cube_2 quat (4D)
        - cube_3 pos (3D) - relative to env origin  
        - cube_3 quat (4D)
        - gripper to cube_1 (3D)
        - gripper to cube_2 (3D)
        - gripper to cube_3 (3D)
        - cube_1 to cube_2 (3D)
        - cube_2 to cube_3 (3D)
        - cube_1 to cube_3 (3D)
        Total: 3+4+3+4+3+4+3+3+3+3+3+3 = 39D
        """

        # Get end-effector position in the world frame
        ee_pos = np.array([
            self.current_eef_pose.pose.position.x,
            self.current_eef_pose.pose.position.y,
            self.current_eef_pose.pose.position.z
        ])

        # Get cube positions and quaternions
        cube_1_pos = self.cube_positions['cube_1'] 
        cube_2_pos = self.cube_positions['cube_2']
        cube_3_pos = self.cube_positions['cube_3']

        # Get cube quaternions in the correct format (w, x, y, z)
        # IsaacLab expects quaternions in [w, x, y, z] format
        cube_1_quat = np.array([
            self.cube_quaternions['cube_1'][0],  
            self.cube_quaternions['cube_1'][1],     
            self.cube_quaternions['cube_1'][2],  
            self.cube_quaternions['cube_1'][3]   
        ])
        
        cube_2_quat = np.array([
            self.cube_quaternions['cube_2'][0],  
            self.cube_quaternions['cube_2'][1],  
            self.cube_quaternions['cube_2'][2],  
            self.cube_quaternions['cube_2'][3]   
        ])
        
        cube_3_quat = np.array([
            self.cube_quaternions['cube_3'][0],  
            self.cube_quaternions['cube_3'][1],  
            self.cube_quaternions['cube_3'][2],  
            self.cube_quaternions['cube_3'][3]   
        ])

        # Compute relative positions from the environment origin
        cube_1_pos_rel = cube_1_pos - self.env_origin
        cube_2_pos_rel = cube_2_pos - self.env_origin
        cube_3_pos_rel = cube_3_pos - self.env_origin

        # Compute gripper to cube vectors
        gripper_to_cube_1 = cube_1_pos - ee_pos
        gripper_to_cube_2 = cube_2_pos - ee_pos
        gripper_to_cube_3 = cube_3_pos - ee_pos

        # Compute cube to cube vectors
        cube_1_to_2 = cube_1_pos - cube_2_pos
        cube_2_to_3 = cube_2_pos - cube_3_pos
        cube_1_to_3 = cube_1_pos - cube_3_pos

        # Concatenate all observations into a single array
        object_obs = np.concatenate([
            cube_1_pos_rel,  # [3]
            cube_1_quat,     # [4]
            cube_2_pos_rel,  # [3]
            cube_2_quat,     # [4]
            cube_3_pos_rel,  # [3]
            cube_3_quat,     # [4]
            gripper_to_cube_1,  # [3]
            gripper_to_cube_2,  # [3]
            gripper_to_cube_3,  # [3]
            cube_1_to_2,    # [3]
            cube_2_to_3,    # [3]
            cube_1_to_3     # [3]
        ])

        return object_obs

    def update_status(self, status: str, additional_info: str = ""):
        """Update status display without interfering with other output"""
        # Simple status update without cursor manipulation
        print(f"\n{status}{additional_info}")
        print()  # Add spacing

    def save_observation_to_csv(self, obs_dict: Dict[str, np.ndarray], action_np: Optional[np.ndarray] = None):
        """
        Save EEF pose observations and actions to CSV file with timestamps for dynamics analysis.
        Saves position (x,y,z), quaternion (x,y,z,w) components, and action data.
        
        Args:
            obs_dict: Observation dictionary containing eef_pos and eef_quat
            action_np: Action array [x, y, z, qw, qx, qy, qz, gripper_cmd] (optional)
        """
        try:
            import csv
            import os
            from datetime import datetime
            
            # Create data directory if it doesn't exist
            data_dir = os.path.join(os.path.expanduser("~"), "bc_policy_data")
            os.makedirs(data_dir, exist_ok=True)
            
            # Generate filename with timestamp if not exists
            if not hasattr(self, 'csv_filename'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.csv_filename = os.path.join(data_dir, f"eef_dynamics_{timestamp}.csv")
                self.csv_file_initialized = False
            
            # Get current timestamp
            current_time = datetime.now()
            timestamp_str = current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # Include milliseconds
            
            # Extract EEF position and quaternion from observation
            eef_pos = obs_dict['eef_pos']      # [x, y, z]
            eef_quat = obs_dict['eef_quat']    # [qw, qx, qy, qz] - IsaacLab format
            
            # Convert quaternion from IsaacLab [qw, qx, qy, qz] to standard [qx, qy, qz, qw] for CSV
            eef_quat_standard = np.array([eef_quat[1], eef_quat[2], eef_quat[3], eef_quat[0]])
            
            # Prepare row data with observation
            row_data = [
                timestamp_str,
                self.step_count,
                float(eef_pos[0]),    # x
                float(eef_pos[1]),    # y  
                float(eef_pos[2]),    # z
                float(eef_quat_standard[0]),  # qx
                float(eef_quat_standard[1]),  # qy
                float(eef_quat_standard[2]),  # qz
                float(eef_quat_standard[3])   # qw
            ]
            
            # Add action data if provided
            if action_np is not None and len(action_np) >= 8:
                # Action format: [x, y, z, qw, qx, qy, qz, gripper_cmd]
                action_pos = action_np[:3]      # [x, y, z]
                action_quat_sim = action_np[3:7] # [qw, qx, qy, qz] - IsaacLab format
                action_gripper = action_np[7]   # gripper command
                
                # Convert action quaternion from IsaacLab [qw, qx, qy, qz] to standard [qx, qy, qz, qw]
                action_quat_standard = np.array([action_quat_sim[1], action_quat_sim[2], action_quat_sim[3], action_quat_sim[0]])
                
                # Add action data to row
                row_data.extend([
                    float(action_pos[0]),         # action_x
                    float(action_pos[1]),         # action_y
                    float(action_pos[2]),         # action_z
                    float(action_quat_standard[0]), # action_qx
                    float(action_quat_standard[1]), # action_qy
                    float(action_quat_standard[2]), # action_qz
                    float(action_quat_standard[3]), # action_qw
                    float(action_gripper)         # action_gripper
                ])
            else:
                # Add empty action columns if no action provided
                row_data.extend([None] * 8)  # 8 action columns
            
            # Write to CSV file
            with open(self.csv_filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header if this is the first time
                if not self.csv_file_initialized:
                    header = [
                        'timestamp',
                        'step_count',
                        'eef_pos_x',
                        'eef_pos_y', 
                        'eef_pos_z',
                        'eef_quat_x',
                        'eef_quat_y',
                        'eef_quat_z',
                        'eef_quat_w',
                        'action_x',
                        'action_y',
                        'action_z',
                        'action_quat_x',
                        'action_quat_y',
                        'action_quat_z',
                        'action_quat_w',
                        'action_gripper'
                    ]
                    writer.writerow(header)
                    self.csv_file_initialized = True
                    self.get_logger().info(f"Created EEF dynamics CSV file with action data: {self.csv_filename}")
                
                # Write data row
                writer.writerow(row_data)
                
        except Exception as e:
            self.get_logger().error(f"Error saving observation to CSV: {e}")

    def check_keyboard_input(self):
        """Check for keyboard input and handle commands"""
        try:
            key = self.keyboard.get_key()
            if key is None:
                return
                
            key = key.lower()
            
            # Force flush before processing commands
            sys.stdout.flush()
            sys.stderr.flush()
            
            if self.replay_mode:
                # ... existing replay mode commands ...
                pass
            else:
                # Normal mode commands
                if key == ' ':  # Space bar - start/resume
                    if not self.is_running:
                        self.start_policy()
                        
                elif key == 's':  # S - stop
                    if self.is_running:
                        self.stop_policy()
                        
                elif key == 'r':  # R - reset episode
                    self.reset_to_home()
                    
                elif key == 'o':  # O - manual gripper toggle
                    self.toggle_gripper_manual()
                    
                elif key == 'c':  # C - randomly spawn cubes
                    self.randomly_spawn_cubes()

                elif key == 'g':  # G - emergency gripper reset
                    with self.gripper_action_lock:
                        self.gripper_action_in_progress = False
                        self.gripper_last_command_time = 0.0
                    print("\n🚨 Emergency gripper reset performed")
                    
                elif key == '1':  # 1 - spawn cubes in line pattern
                    self.spawn_cubes_in_pattern("line")
                    
                elif key == '2':  # 2 - spawn cubes in triangle pattern
                    self.spawn_cubes_in_pattern("triangle")
                    
                elif key == '3':  # 3 - spawn cubes in stack-ready pattern
                    self.spawn_cubes_in_pattern("stack_ready")
                
                # PRESET COMMANDS (Numbers 4-9, 0)
                elif key == '4':  # 4 - default preset
                    self.spawn_cubes_preset("default")
                    
                elif key == '5':  # 5 - custom_1 preset (your requested positions)
                    self.spawn_cubes_preset("custom_1")
                    
                elif key == '6':  # 6 - wide_spread preset
                    self.spawn_cubes_preset("wide_spread")
                    
                elif key == '7':  # 7 - tight_cluster preset
                    self.spawn_cubes_preset("tight_cluster")
                    
                elif key == '8':  # 8 - corner_formation preset
                    self.spawn_cubes_preset("corner_formation")
                    
                elif key == '9':  # 9 - stacking_ready preset
                    self.spawn_cubes_preset("stacking_ready")
                    
                elif key == '0':  # 0 - manipulation_test preset
                    self.spawn_cubes_preset("manipulation_test")
                
                # NEW LETTER COMMANDS FOR REMAINING PRESETS
                elif key == 'a':  # A - reach_challenge preset
                    self.spawn_cubes_preset("reach_challenge")
                    
                elif key == 'b':  # B - pick_place_demo preset
                    self.spawn_cubes_preset("pick_place_demo")
                    
                elif key == 'd':  # D - sorting_task preset
                    self.spawn_cubes_preset("sorting_task")
                    
                elif key == 'e':  # E - assembly_line preset
                    self.spawn_cubes_preset("assembly_line")
                    
                elif key == 'f':  # F - circular_arrangement preset
                    self.spawn_cubes_preset("circular_arrangement")
                    
                elif key == 'h':  # H - precision_test preset
                    self.spawn_cubes_preset("precision_test")
                    
                elif key == 'i':  # I - learning_progression_1 preset
                    self.spawn_cubes_preset("learning_progression_1")
                    
                elif key == 'j':  # J - learning_progression_2 preset
                    self.spawn_cubes_preset("learning_progression_2")
                    
                elif key == 'k':  # K - workspace_corners preset
                    self.spawn_cubes_preset("workspace_corners")
                    
                elif key == 'l':  # L - list all presets
                    self.list_cube_presets()
                
                elif key == 'q':  # Q - quit
                    self.shutdown_requested = True
                    print("\nShutdown requested...")
                    sys.stdout.flush()
                    raise KeyboardInterrupt("User requested shutdown")
                    
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"\nKeyboard input error: {e}")
            sys.stdout.flush()

    # Show trial information
    def show_trial_info(self):
        """Display information about current trial"""
        if not self.replay_mode or not self.replay_data:
            return
            
        current_trial = self.replay_data[self.replay_trial]
        metadata = current_trial['metadata']
        observations = current_trial['observations']
        
        print(f"\n{'='*60}")
        print(f"TRIAL {self.replay_trial} INFORMATION")
        print(f"{'='*60}")
        print(f"Task:           {metadata['task']}")
        print(f"Checkpoint:     {os.path.basename(metadata['checkpoint'])}")
        print(f"Horizon:        {metadata['horizon']}")
        print(f"Frequency:      {metadata['frequency_hz']} Hz")
        print(f"dt:             {metadata['dt']} seconds")
        print(f"Total steps:    {metadata['total_steps']}")
        print(f"Observations:   {len(observations)}")
        print(f"Current step:   {self.replay_index}")
        print(f"Progress:       {self.replay_index}/{len(observations)-1} ({100*self.replay_index/max(1,len(observations)-1):.1f}%)")
        print(f"{'='*60}")

    def toggle_gripper_manual(self):
        """Manually toggle gripper state between open and closed"""
        try:
            if self.gripper_goal_state == 'open' or self.gripper_goal_state == 'unknown':
                # Close the gripper
                self.close_gripper()
                print("Manual gripper command: CLOSING")
                self.update_status("Manual gripper: CLOSING", " - Press O again to open")
                
            elif self.gripper_goal_state == 'closed':
                # Open the gripper
                self.open_gripper()
                print("Manual gripper command: OPENING")
                self.update_status("Manual gripper: OPENING", " - Press O again to close")
                
        except Exception as e:
            self.get_logger().error(f"Error in manual gripper control: {e}")
            print(f"Manual gripper control failed: {e}")

    def start_policy(self):
        """Start the policy with proper LSTM reset"""
        if not self.is_running:
            self.is_running = True
            self.episode_active = True
            self.step_count = 0
            
            # Always reset LSTM state when starting new policy run
            self.policy.start_episode()
            
            self.update_status("Policy started", f"Control freq: {self.control_frequency}Hz")

    def stop_policy(self):
        """Stop the policy and reset for next episode"""
        if self.is_running:
            self.is_running = False
            self.episode_active = False
            
            # Reset LSTM state when stopping
            self.policy.start_episode()
            
            self.update_status("Policy stopped")

    def reset_to_home(self):
        """Reset robot to safe home position"""
        # Stop policy execution first
        was_running = self.is_running
        if self.is_running:
            self.stop_policy()
        
        print("Resetting robot to home position...")
        self.update_status("Status: HOMING - Moving to safe position...")
        
        try:
            # Define safe home position (adjust these values based on your robot setup)
            home_position = np.array([0.46, 0.0, 0.266])  # Safe position above workspace
            home_quaternion_ros = np.array([1.0, 0.0, 0.0, 0.0])  # Pointing down [qx, qy, qz, qw]
            
            # Create cartesian pose command: [x, y, z, qx, qy, qz, qw]
            home_pose = np.concatenate([
                home_position,        # [x, y, z]
                home_quaternion_ros   # [qx, qy, qz, qw]
            ])
            
            # Publish home pose command
            pose_msg = Float64MultiArray()
            pose_msg.data = home_pose.tolist()
            self.pose_command_pub.publish(pose_msg)
            
            # Open gripper to safe state
            self.open_gripper()
            
            # Reset episode state
            self.policy.start_episode()
            self.gripper_goal_state = 'open'
            
            # RESET CUBE ATTACHMENT STATE
            self.grasp_sequence_count = 0
            self.cube_attached = None
            self.last_gripper_state = 'open'
            self.gripper_state_stable_count = 0
            print("🔄 Cube attachment state reset")
            
            print("Robot moved to home position and episode state reset")
            
            # Update status based on previous running state
            if was_running:
                self.update_status("Status: HOMED - Press SPACE to resume, S to stop, Q to quit")
            else:
                self.update_status("Status: HOMED - Press SPACE to start, Q to quit")
                
        except Exception as e:
            self.get_logger().error(f"Error during home reset: {e}")
            self.update_status("Status: HOME FAILED - Check robot state")
    
    # Create observation dictionary for the policy x_t -> Input to the policy
    def create_observation(self) -> Optional[Dict[str, np.ndarray]]:
        """Create observation dictionary from current robot state for robomimic policy"""
        # Extract end-effector position
        eef_pos = np.array([
            self.current_eef_pose.pose.position.x,
            self.current_eef_pose.pose.position.y,
            self.current_eef_pose.pose.position.z
        ], dtype=np.float32)
        
        # Extract end-effector quaternion from ROS message [qx, qy, qz, qw]
        eef_quat_ros = np.array([
            self.current_eef_pose.pose.orientation.x,  # qx
            self.current_eef_pose.pose.orientation.y,  # qy  
            self.current_eef_pose.pose.orientation.z,  # qz
            self.current_eef_pose.pose.orientation.w   # qw
        ], dtype=np.float32)
        
        # TRANSFORM: Convert from ROS [qx, qy, qz, qw] to IsaacLab [qw, qx, qy, qz] for policy input
        eef_quat_sim = np.array([
            eef_quat_ros[3],  # qw
            eef_quat_ros[0],  # qx
            eef_quat_ros[1],  # qy
            eef_quat_ros[2]   # qz
        ], dtype=np.float32)
        
        # Extract gripper positions 
        gripper_pos = self.current_gripper_positions.astype(np.float32)
        
        # Compute object observations
        object_state = self.compute_object_observations().astype(np.float32)
        
        # Return dictionary with proper keys for robomimic policy (numpy arrays)
        obs_dict = {
            'eef_pos': eef_pos,
            'eef_quat': eef_quat_sim,
            'gripper_pos': gripper_pos,
            'object': object_state
        }
        
        return obs_dict
    
    def log_observation_compact(self, obs_dict: Dict[str, np.ndarray], action_np: np.ndarray = None):
        """Compact structured observation logging for full 48D vector with optional action logging"""
        eef_pos = obs_dict['eef_pos']           # 3D
        eef_quat = obs_dict['eef_quat']         # 4D  
        gripper_pos = obs_dict['gripper_pos']   # 2D
        object_obs = obs_dict['object']         # 39D
        
        # Calculate total observation size
        total_size = len(eef_pos) + len(eef_quat) + len(gripper_pos) + len(object_obs)
        
        print(f"\n┌{'─'*100}┐")
        print(f"│ STEP {self.step_count:<6} │ FULL 48D OBSERVATION VECTOR ({total_size} elements) │")
        print(f"├{'─'*100}┤")
        
        # End-Effector Position (elements 0-2)
        print(f"│ EEF POS [0-2]   │ X:{eef_pos[0]:8.5f} │ Y:{eef_pos[1]:8.5f} │ Z:{eef_pos[2]:8.5f} │")
        
        # End-Effector Quaternion (elements 3-6)
        print(f"│ EEF QUAT [3-6]  │ X:{eef_quat[0]:8.5f} │ Y:{eef_quat[1]:8.5f} │ Z:{eef_quat[2]:8.5f} │ W:{eef_quat[3]:8.5f} │")
        
        # Gripper Position (elements 7-8)
        gripper_width = abs(gripper_pos[0]) + abs(gripper_pos[1])
        gripper_state = "OPEN" if gripper_width > 0.04 else "CLOSED"
        print(f"│ GRIPPER [7-8]   │ F1:{gripper_pos[0]:8.5f} │ F2:{gripper_pos[1]:8.5f} │ Width:{gripper_width:7.4f} │ {gripper_state:<6} │")

        # Add this after the gripper status line in log_observation_compact:

        # Add attachment status display
        attachment_status = f"ATTACHED: {self.cube_attached}" if self.cube_attached else "ATTACHED: None"
        grasp_info = f"Grasp Seq: {self.grasp_sequence_count}/2"
        print(f"│ ATTACHMENT      │ {attachment_status:<20} │ {grasp_info:<15} │ State: {self.last_gripper_state:<6} │")
        
        print(f"├{'─'*100}┤")
        print(f"│ OBJECT STATE [9-47] - 39 ELEMENTS │")
        print(f"├{'─'*100}┤")
        
        # Parse object observations (39D breakdown)
        idx = 0
        
        # Cube 1 Position + Quaternion (elements 9-15)
        cube1_pos = object_obs[idx:idx+3]
        cube1_quat = object_obs[idx+3:idx+7]
        print(f"│ CUBE 1 [9-15]   │ Pos: [{cube1_pos[0]:6.3f}, {cube1_pos[1]:6.3f}, {cube1_pos[2]:6.3f}] │ Quat: [{cube1_quat[0]:5.2f}, {cube1_quat[1]:5.2f}, {cube1_quat[2]:5.2f}, {cube1_quat[3]:5.2f}] │")
        idx += 7
        
        # Cube 2 Position + Quaternion (elements 16-22)
        cube2_pos = object_obs[idx:idx+3]
        cube2_quat = object_obs[idx+3:idx+7]
        print(f"│ CUBE 2 [16-22]  │ Pos: [{cube2_pos[0]:6.3f}, {cube2_pos[1]:6.3f}, {cube2_pos[2]:6.3f}] │ Quat: [{cube2_quat[0]:5.2f}, {cube2_quat[1]:5.2f}, {cube2_quat[2]:5.2f}, {cube2_quat[3]:5.2f}] │")
        idx += 7
        
        # Cube 3 Position + Quaternion (elements 23-29)
        cube3_pos = object_obs[idx:idx+3]
        cube3_quat = object_obs[idx+3:idx+7]
        print(f"│ CUBE 3 [23-29]  │ Pos: [{cube3_pos[0]:6.3f}, {cube3_pos[1]:6.3f}, {cube3_pos[2]:6.3f}] │ Quat: [{cube3_quat[0]:5.2f}, {cube3_quat[1]:5.2f}, {cube3_quat[2]:5.2f}, {cube3_quat[3]:5.2f}] │")
        idx += 7
        
        # Relative positions EEF to Cubes (elements 30-38)
        print(f"├{'─'*100}┤")
        print(f"│ EEF TO CUBE RELATIVE POSITIONS │")
        print(f"├{'─'*100}┤")
        for i in range(3):
            rel_pos = object_obs[idx:idx+3]
            distance = np.linalg.norm(rel_pos)
            element_range = f"[{30+i*3}-{32+i*3}]"
            print(f"│ EEF→CUBE{i+1} {element_range} │ Rel: [{rel_pos[0]:7.4f}, {rel_pos[1]:7.4f}, {rel_pos[2]:7.4f}] │ Dist: {distance:6.4f}m │")
            idx += 3
        
        # Cube-to-Cube relative positions (elements 39-47)
        print(f"├{'─'*100}┤")
        print(f"│ CUBE TO CUBE RELATIVE POSITIONS │")
        print(f"├{'─'*100}┤")
        
        # Cube 1 to Cube 2 (elements 39-41)
        cube1_to_cube2 = object_obs[idx:idx+3]
        distance_1_2 = np.linalg.norm(cube1_to_cube2)
        print(f"│ CUBE1→CUBE2 [39-41] │ Rel: [{cube1_to_cube2[0]:7.4f}, {cube1_to_cube2[1]:7.4f}, {cube1_to_cube2[2]:7.4f}] │ Dist: {distance_1_2:6.4f}m │")
        idx += 3
        
        # Cube 2 to Cube 3 (elements 42-44)
        cube2_to_cube3 = object_obs[idx:idx+3]
        distance_2_3 = np.linalg.norm(cube2_to_cube3)
        print(f"│ CUBE2→CUBE3 [42-44] │ Rel: [{cube2_to_cube3[0]:7.4f}, {cube2_to_cube3[1]:7.4f}, {cube2_to_cube3[2]:7.4f}] │ Dist: {distance_2_3:6.4f}m │")
        idx += 3
        
        # Cube 1 to Cube 3 (elements 45-47)
        cube1_to_cube3 = object_obs[idx:idx+3]
        distance_1_3 = np.linalg.norm(cube1_to_cube3)
        print(f"│ CUBE1→CUBE3 [45-47] │ Rel: [{cube1_to_cube3[0]:7.4f}, {cube1_to_cube3[1]:7.4f}, {cube1_to_cube3[2]:7.4f}] │ Dist: {distance_1_3:6.4f}m │")
        idx += 3
        
        # Action logging section (if action is provided)
        if action_np is not None:
            print(f"├{'─'*100}┤")
            print(f"│ POLICY ACTION OUTPUT - 8D ACTION VECTOR │")
            print(f"├{'─'*100}┤")
            
            # Ensure it's a 1D array
            if action_np.ndim > 1:
                action_np = action_np.squeeze()
            
            # Parse action components
            eef_pose_action = action_np[:7]  # [x, y, z, qw, qx, qy, qz] - IsaacLab format
            gripper_action = action_np[7]    # Gripper command
            
            # End-effector pose action
            action_pos = eef_pose_action[:3]
            action_quat = eef_pose_action[3:]  # [qw, qx, qy, qz]
            
            print(f"│ ACTION POS [0-2] │ X:{action_pos[0]:8.5f} │ Y:{action_pos[1]:8.5f} │ Z:{action_pos[2]:8.5f} │")
            print(f"│ ACTION QUAT[3-6] │ W:{action_quat[0]:8.5f} │ X:{action_quat[1]:8.5f} │ Y:{action_quat[2]:8.5f} │ Z:{action_quat[3]:8.5f} │")
            
            # Gripper action analysis
            gripper_cmd_clamped = np.clip(gripper_action, -1.0, 1.0)
            gripper_pos_cmd = (gripper_cmd_clamped + 1.0) * 0.04  # Maps [-1,1] to [0, 0.08]
            gripper_state_cmd = "CLOSE" if gripper_action < 0 else "OPEN"
            
            print(f"│ ACTION GRIP [7]  │ Raw:{gripper_action:8.5f} │ Clamped:{gripper_cmd_clamped:8.5f} │ Pos:{gripper_pos_cmd:7.4f} │ Cmd:{gripper_state_cmd:<6} │")
            
            # Action magnitude analysis
            pos_change_mag = np.linalg.norm(action_pos - eef_pos)
            quat_diff = np.abs(action_quat - eef_quat).sum()
            
            print(f"├{'─'*100}┤")
            print(f"│ ACTION ANALYSIS │ Pos Change: {pos_change_mag:6.4f}m │ Quat Diff: {quat_diff:6.4f} │ Gripper Δ: {gripper_action:7.4f} │")
        
        # Summary statistics
        print(f"├{'─'*100}┤")
        eef_to_cubes_min_dist = min([np.linalg.norm(object_obs[21+i*3:24+i*3]) for i in range(3)])
        cube_to_cube_min_dist = min([distance_1_2, distance_2_3, distance_1_3])
        print(f"│ SUMMARY         │ Total: {total_size} elements │ EEF height: {eef_pos[2]:6.4f}m │ Closest EEF→Cube: {eef_to_cubes_min_dist:6.4f}m │ Closest Cube→Cube: {cube_to_cube_min_dist:6.4f}m │")
        print(f"└{'─'*100}┘")
        # Increment step count
        self.step_count += 1
    
    def unified_control_loop(self):
        """Unified control loop with consistent timing for both normal and replay modes"""
        if self.shutdown_requested:
            return
            
        try:
            if self.replay_mode and self.replay_auto:
                # Handle replay mode execution
                self.handle_replay_step()
            elif not self.replay_mode and self.is_running and self.episode_active:
                # Handle normal mode execution
                self.handle_normal_step()
        except Exception as e:
            self.get_logger().error(f"Unified control loop error: {e}")

    def handle_normal_step(self):
        """Handle normal mode execution with proper timing"""
        try:
            # STEP 1: Check for gripper state changes and handle cube attachment
            gripper_state_change = self.detect_gripper_state_change()
            if gripper_state_change:
                self.handle_cube_attachment(gripper_state_change)
            
            # STEP 2: Update attached cube pose if any cube is attached
            self.update_attached_cube_pose()
            
            # STEP 3: Create observation for the policy (with updated cube poses)
            obs_dict = self.create_observation()
            if obs_dict is not None:
                
                # STEP 4: Run policy inference
                action = self.policy(obs_dict)
                action_np = action if isinstance(action, np.ndarray) else action.cpu().numpy()

                # STEP 5: Save observation for the normal policy inference
                self.save_observation_to_csv(obs_dict, action_np)
                
                # STEP 6: Log observation and action together
                self.log_observation_compact(obs_dict, action_np)
                
                # STEP 7: Execute the action
                self.execute_action(action_np)

        except Exception as e:
            self.get_logger().error(f"Normal mode execution error: {e}")

    def handle_replay_step(self):
        """Handle replay mode execution with proper timing"""
        if not self.replay_mode or not self.replay_data:
            return
        
        try:
            # Execute one replay step - no manual timing checks needed
            self.replay_step()
        except Exception as e:
            self.get_logger().error(f"Replay mode execution error: {e}")

    def cleanup(self):
        """Cleanup resources"""
        try:
            self.keyboard.restore_terminal()
        except:
            pass
        
        # Destroy action clients (from policy_runner.py)
        try:
            self.homing_client.destroy()
            self.move_client.destroy()
            self.grasp_client.destroy()
            self.get_logger().info("Gripper action clients destroyed.")
        except:
            pass

    def publish_observation_debug(self, obs_dict: Dict[str, np.ndarray]):
        """Publish observation for debugging purposes"""
        try:
            # Concatenate all observation components into a single array (same as policy input)
            obs_flat = np.concatenate([
                obs_dict['eef_pos'],      # [3]
                obs_dict['eef_quat'],     # [4]  
                obs_dict['gripper_pos'],  # [2]
                obs_dict['object']        # [39]
            ])  # Result: [48]
            
            # Create and publish message
            obs_msg = Float64MultiArray()
            obs_msg.data = obs_flat.tolist()
            self.observation_debug_pub.publish(obs_msg)
                
        except Exception as e:
            self.get_logger().warn(f"Error publishing observation debug: {e}")
        

class KeyboardInput:
    """Handles keyboard input in a non-blocking way"""
    
    def __init__(self):
        self.old_settings = None
        self.setup_terminal()
        
    def setup_terminal(self):
        """Setup terminal for non-blocking input"""
        try:
            self.old_settings = termios.tcgetattr(sys.stdin)
            new_settings = termios.tcgetattr(sys.stdin)
            
            # Use cbreak mode instead of raw mode for better formatting
            new_settings[3] = new_settings[3] & ~(termios.ECHO | termios.ICANON)
            new_settings[6][termios.VMIN] = 0  # Non-blocking read
            new_settings[6][termios.VTIME] = 0  # No timeout
            
            termios.tcsetattr(sys.stdin, termios.TCSANOW, new_settings)
        except:
            pass
        
    def restore_terminal(self):
        """Restore terminal settings"""
        try:
            if self.old_settings:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
                # Force flush after restoring terminal
                sys.stdout.flush()
                sys.stderr.flush()
        except:
            pass
        
    def get_key(self):
        """Get a single keypress if available"""
        try:
            if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                key = sys.stdin.read(1)
                # Force flush after reading key to prevent formatting issues
                sys.stdout.flush()
                return key
            return None
        except:
            return None

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="BC Policy Runner for Franka Robot")
    parser.add_argument("--policy", type=str, required=True,
                       help="Path to the trained BC policy file (.pth)")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to run inference on (cpu or cuda)")
    parser.add_argument("--deterministic", action="store_true",
                       help="Use deterministic policy inference")
    parser.add_argument("--frequency", type=float, default=20.0,
                       help="Control frequency in Hz")
    # NEW: Add replay mode argument
    parser.add_argument("--replay", type=str, default=None,
                       help="Path to JSON file with recorded observations for replay mode")
    
    args = parser.parse_args()
    
    # Initialize ROS2
    rclpy.init()
    
    node = None
    try:
        node = BCPolicyRunner(
            policy_path=args.policy,
            device=args.device,
            deterministic=args.deterministic,
            control_frequency=args.frequency,
            replay_file=args.replay  # NEW: Pass replay file
        )
        
        # Use executor with timeout to allow KeyboardInterrupt handling
        executor = rclpy.executors.MultiThreadedExecutor()
        executor.add_node(node)
        
        try:
            executor.spin()
        except KeyboardInterrupt:
            print("\n\nCTRL+C detected - Shutting down BC Policy Runner...")
        finally:
            executor.shutdown()
        
    except KeyboardInterrupt:
        print("\n\nCTRL+C detected during initialization - Shutting down...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup node
        if node is not None:
            try:
                node.cleanup()
                node.destroy_node()
            except:
                pass
        
        # Shutdown ROS2
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except:
            pass
        
        print("Shutdown complete.")

if __name__ == "__main__":
    main()