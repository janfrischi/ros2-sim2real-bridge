#!/usr/bin/env python3
"""
Behavior Cloning Policy Runner for Franka Robot
This node loads a trained BC LSTM+GMM policy and runs inference on the Franka robot.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional
import argparse
import sys
import termios
import tty
import select

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from franka_msgs.action import Homing, Move, Grasp
from action_msgs.msg import GoalStatus
import tf2_ros

# Define the LSTMGMMNetwork class
class LSTMGMMNetwork(nn.Module):
    """LSTM + GMM Actor Network for Behavior Cloning - IsaacLab Compatible"""
    
    def __init__(self, obs_dim: int = 48, action_dim: int = 8, hidden_dim: int = 400, 
                 num_layers: int = 2, num_modes: int = 5, min_std: float = 0.0001,
                 std_activation: str = "softplus", low_noise_eval: bool = True):
        super().__init__()  # Inherit from nn.Module

        self.obs_dim = obs_dim # Dimension of the observation space
        self.action_dim = action_dim # Dimension of the action space "Absolute end-effector pose + gripper command"
        self.hidden_dim = hidden_dim # Dimension of LSTM hidden state
        self.num_layers = num_layers # Number of LSTM layers
        self.num_modes = num_modes # Number of Gaussians in GMM (same as num_nodes in IsaacLab)
        self.min_std = min_std # Minimum standard deviation for Gaussian scale parameters
        self.std_activation = std_activation # Activation function for standard deviation
        self.low_noise_eval = low_noise_eval # Flag for low noise evaluation
        
        # Architecture for LSTM Network
        self.lstm = nn.LSTM(
            input_size=obs_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False  # Explicitly set from config
        )

        # Architecture for Gaussian Mixture Model (GMM) heads
        self.gmm = nn.ModuleDict({
            'mean': nn.Linear(hidden_dim, num_modes * action_dim),  
            'scale': nn.Linear(hidden_dim, num_modes * action_dim),
            'logits': nn.Linear(hidden_dim, num_modes) 
            })

        # Hidden states for sequential inference
        self.hidden_states = None
    
    # Method for resetting LSTM hidden states, when starting a new episode
    def reset_hidden_states(self, batch_size: int = 1):
        """Reset LSTM hidden states"""
        device = next(self.parameters()).device
        self.hidden_states = (
            torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device),
            torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        )
    
    # Define the forward pass for the network
    def forward(self, obs: Dict[str, torch.Tensor], deterministic: bool = True) -> torch.Tensor:
        """
        Forward pass through the network
        Args:
            obs: Dictionary of observation tensors with keys ['eef_pos', 'eef_quat', 'gripper_pos', 'object']
            deterministic: If True, return mean of most likely mode
        Returns:
            action: Sampled action [batch_size, action_dim]
        """
        # Concatenate dictionary observations into a single tensor
        obs_tensor = torch.cat([
            obs['eef_pos'],      # [batch_size, seq_length, 3]
            obs['eef_quat'],     # [batch_size, seq_length, 4]  
            obs['gripper_pos'],  # [batch_size, seq_length, 2]
            obs['object']        # [batch_size, seq_length, 39]
        ], dim=-1)  # Result: [batch_size, seq_length, 48]
        
        # No need to add sequence dimension - it should already be there
        batch_size, seq_len, _ = obs_tensor.shape
        
        # Initialize hidden states if needed
        if self.hidden_states is None:
            self.reset_hidden_states(batch_size)
            
        # LSTM forward pass -> Returns two values
        # lstm_out: [batch_size, seq_length, hidden_dim]
        # self.hidden_states: tuple of (h_n, c_n) with shape [num_layers, batch_size, hidden_dim]
        lstm_out, self.hidden_states = self.lstm(obs_tensor, self.hidden_states)
        
        # Take the last timestep output (h_t) hidden state vector
        h_t = lstm_out[:, -1, :]  # [batch_size, hidden_dim]
        
        # Pass h_t through GMM heads to get means, scales, and logits of the Gaussian Mixture Model
        means = self.gmm['mean'](h_t).view(batch_size, self.num_modes, self.action_dim)  # Reshape to [batch_size, num_modes, action_dim]
        scales = nn.functional.softplus(self.gmm['scale'](h_t)).view(batch_size, self.num_modes, self.action_dim)  # Reshape to [batch_size, num_modes, action_dim]
        scales = torch.clamp(scales, min=self.min_std)
        logits = self.gmm['logits'](h_t)  # [batch_size, num_modes]

        # Apply low noise evaluation if enabled
        if self.low_noise_eval and not self.training:
            scales = scales * 0.1  # Reduce noise during evaluation

        # Handle deterministic vs stochastic action selection from the Gaussian Mixture model
        if deterministic:
            # Return mean of most likely mode
            mode_probs = torch.softmax(logits, dim=-1) # Run softmax on logits to get probabilities
            best_mode = torch.argmax(mode_probs, dim=-1)
            # Get the means for the best mode, actions contains the 8d mean vector of the most probable Gaussian mode
            actions = means[torch.arange(batch_size), best_mode]
            return actions
        
        else:
            # Stochastic Sampling from GMM

            # Step 1. Sample a mode from the GMM
            mode_probs = torch.softmax(logits, dim=-1)
            mode_dist = torch.distributions.Categorical(mode_probs) # Create a categorical distribution over modes
            selected_modes = mode_dist.sample() # Randomly select Gaussian mode
            
            # Step 2. Sample actions from the selected Gaussian
            selected_means = means[torch.arange(batch_size), selected_modes]
            selected_scales = scales[torch.arange(batch_size), selected_modes]

            # Step 3. Create the Gaussian Distribution
            gaussian_dist = torch.distributions.Normal(selected_means, selected_scales)
            # Step 4. Sample actions from the Gaussian distribution
            actions = gaussian_dist.sample()
            return actions

# BCPolicy Runner Node
class BCPolicyRunner(Node):
    """ROS2 Node for running Behavior Cloning policy on Franka robot"""
    # We use 20Hz control frequency as this was the default in the original IsaacLab implementation
    def __init__(self, policy_path: str, device: str = "cpu", deterministic: bool = True, 
                 control_frequency: float = 20.0):
        super().__init__('bc_policy_runner')
        
        # Initialize parameters
        self.device = torch.device(device)
        self.deterministic = deterministic
        self.control_frequency = control_frequency
        
        # Load policy and set it to evaluation mode
        self.policy = self.load_policy(policy_path)
        self.policy.eval()
        
        # Robot state storage
        self.current_eef_pose = None
        self.current_gripper_positions = None

        # Add sequence buffer for RNN
        self.seq_length = 10  # Match config
        self.observation_buffer = []
        
        # Zero vector test mode
        self.zero_vector_mode = False
        
        # Object state storage - hardcoded for now
        self.cube_positions = {
            'cube_1': np.array([0.5, 0.2, 0.0203]),
            'cube_2': np.array([0.5, -0.2, 0.0203]),
            'cube_3': np.array([0.5, -0.2, 0.0203])
        }
        
        # Cube orientations (quaternions) - w, x, y, z format "IsaacLab expects quaternions in [w, x, y, z] format"
        self.cube_quaternions = {
            'cube_1': np.array([0.0, 0.0, 0.0, 1.0]),  # Identity quaternion
            'cube_2': np.array([0.0, 0.0, 0.0, 1.0]),  # Identity quaternion
            'cube_3': np.array([0.0, 0.0, 0.0, 1.0])   # Identity quaternion
        }

        # Environment origin (base frame reference)
        self.env_origin = np.array([0.0, 0.0, 0.0])

        # Control flags
        self.is_running = False
        self.episode_active = False
        self.shutdown_requested = False
        
        # Keyboard input handler
        self.keyboard = KeyboardInput()
        
        # Callback group for allowing concurrent callbacks
        self.callback_group = ReentrantCallbackGroup()
        
        # --- Gripper Control Initialization (from policy_runner.py) ---
        self.gripper_goal_state = 'unknown' # 'open', 'closed', 'unknown'
        self.gripper_max_width = 0.08 # Max width for Franka Hand
        self.gripper_speed = 0.05 # Default speed (m/s)
        self.gripper_force = 30.0 # Default grasp force (N)
        self.gripper_epsilon_inner = 0.05 # Tolerance for successful grasp
        self.gripper_epsilon_outer = 0.05
        self.object_grasped = False # Flag to indicate if object is grasped

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
        
        # Setup QoS (Quality of Service) profiles
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=1
        )
        
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
        
        # TF2 setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Control timer - control loop runs at specified frequency
        self.control_timer = self.create_timer(
            1.0 / self.control_frequency,
            self.control_loop,
            callback_group=self.callback_group
        )
        
        # Keyboard input timer (check for keypress every 50ms)
        self.keyboard_timer = self.create_timer(0.05, self.check_keyboard_input)
        
        # Print initial instructions
        self.print_instructions()

    # Helper to wait for action servers (from policy_runner.py)
    def wait_for_action_server(self, client, name):
        self.get_logger().info(f'Waiting for {name} action server...')
        while not client.wait_for_server(timeout_sec=2.0) and rclpy.ok():
            self.get_logger().info(f'{name} action server not available, waiting again...')
        if rclpy.ok():
            self.get_logger().info(f'{name} action server found.')
        else:
             self.get_logger().error(f'ROS shutdown while waiting for {name} server.')
             raise SystemExit('ROS shutdown')

    # Method to home the gripper 
    def home_gripper(self):
        goal_msg = Homing.Goal()
        # Send goal async and forget (or handle future if needed)
        self.homing_client.send_goal_async(goal_msg)
        self.gripper_goal_state = 'open' # Assume homing opens the gripper

    def open_gripper(self):
        """Open the gripper using the action client (from policy_runner.py)"""
        goal_msg = Move.Goal()
        goal_msg.width = self.gripper_max_width
        goal_msg.speed = self.gripper_speed
        self.move_client.send_goal_async(goal_msg)
        self.gripper_goal_state = 'open'
        self.object_grasped = False

    def close_gripper(self):
        """Close the gripper using the action client (from policy_runner.py)"""
        goal_msg = Grasp.Goal()
        goal_msg.width = 0.0
        goal_msg.speed = self.gripper_speed
        goal_msg.force = self.gripper_force
        goal_msg.epsilon.inner = self.gripper_epsilon_inner
        goal_msg.epsilon.outer = self.gripper_epsilon_outer

        # Send the goal and register the callback for the result
        grasp_future = self.grasp_client.send_goal_async(goal_msg)
        grasp_future.add_done_callback(self.grasp_goal_response_callback)

        self.gripper_goal_state = 'closed'

    def grasp_goal_response_callback(self, future):
        """Callback when grasp goal response is received (from policy_runner.py)"""
        # The future object contains the result of the goal request
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Grasp goal rejected.')
            return

        self.get_logger().info('Grasp goal accepted. Waiting for result...')
        result_future = goal_handle.get_result_async()
        # Add a callback to be called when the task is done
        result_future.add_done_callback(self.grasp_result_callback)

    # Callback when the grasp action completes (from policy_runner.py)
    def grasp_result_callback(self, future):
        """Callback triggered when the grasp action completes."""
        status = future.result().status
        self.get_logger().info(f"Grasp action completed with status: {status}")

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info("Grasp successful: Object is attached to the ee.")
            self.object_grasped = True
            self.gripper_goal_state = 'closed'
        else: 
            self.get_logger().warning("Grasp failed: Object is not attached to the ee.")
            self.object_grasped = False
        
    def eef_pose_callback(self, msg: PoseStamped):
        """Callback for end-effector pose updates"""
        self.current_eef_pose = msg # Quaternion is in x, y, z, w format
    
    def gripper_state_callback(self, msg: JointState):
        """Callback for gripper state updates"""
        # Gripper should have symmetric but opposite values: [+value, -value]
        finger_1_pos = msg.position[0]  # First finger (positive)
        finger_2_pos = -msg.position[1] if msg.position[1] > 0 else msg.position[1]  # Second finger (negative)
        self.current_gripper_positions = np.array([finger_1_pos, finger_2_pos])

    def print_instructions(self):
        """Print keyboard control instructions"""
        # Clear screen and print instructions with proper formatting
        print("=" * 80)
        print("BC POLICY RUNNER - KEYBOARD CONTROLS".center(80))
        print("=" * 80)
        print("SPACE BAR: Start/Resume policy execution")
        print("S:         Stop policy execution")
        print("R:         Reset episode (clear LSTM hidden states)")
        print("Z:         Toggle Zero Vector Mode (48D zeros input)")
        print("Q:         Quit the program")
        print("=" * 80)
        
        # Display zero vector mode status
        zero_status = "ENABLED" if self.zero_vector_mode else "DISABLED"
        print(f"Zero Vector Mode: {zero_status}".center(80))
        print("-" * 80)
        
        # Display current object positions and quaternions
        print("CURRENT OBJECT STATE".center(80))
        print("-" * 80)
        
        cube_names = {
            'cube_1': 'Blue Cube',
            'cube_2': 'Red Cube',
            'cube_3': 'Green Cube'
        }
        
        for cube_id, cube_name in cube_names.items():
            pos = self.cube_positions[cube_id]
            quat = self.cube_quaternions[cube_id]
            
            print(f"{cube_name:12} ({cube_id}):")
            print(f"  Position:    [{pos[0]:+7.4f}, {pos[1]:+7.4f}, {pos[2]:+7.4f}]")
            print(f"  Quaternion:  [{quat[0]:+7.4f}, {quat[1]:+7.4f}, {quat[2]:+7.4f}, {quat[3]:+7.4f}] (w,x,y,z)")
            print()
        
        # Environment origin
        origin = self.env_origin
        print(f"Environment Origin: [{origin[0]:+7.4f}, {origin[1]:+7.4f}, {origin[2]:+7.4f}]")
        print("-" * 80)
        
        print("Status: STOPPED - Press SPACE to start".center(80))
        print("=" * 80)
        print()  # Add blank line

    def load_policy(self, policy_path: str) -> LSTMGMMNetwork:
        """Load the trained BC policy from checkpoint"""
        try:
            # Load the checkpoint file
            checkpoint = torch.load(policy_path, map_location=self.device)
            # Extract the model weights dictionary
            state_dict = checkpoint['model']
        
            # Create network with config-compatible parameters
            policy = LSTMGMMNetwork(
                obs_dim=48,
                action_dim=8,
                hidden_dim=400,
                num_layers=2,
                num_modes=5,
                min_std=0.0001,
                std_activation="softplus",
                low_noise_eval=True
            ).to(self.device)
            
            # Create a mapping for the weights (only load the ACTIVE components)
            policy_state_dict = {}
            for key, value in state_dict.items():
                # LSTM Components 
                if key.startswith('policy.nets.rnn.nets.'):
                    # Map RNN weights: policy.nets.rnn.nets.* -> lstm.*
                    new_key = key.replace('policy.nets.rnn.nets.', 'lstm.')
                    policy_state_dict[new_key] = value
                # Per-Step GMM Components
                elif key.startswith('policy.nets.rnn.per_step_net.nets.'):
                    # Map per-step weights (ACTIVE PATH): policy.nets.rnn.per_step_net.nets.* -> gmm.*
                    new_key = key.replace('policy.nets.rnn.per_step_net.nets.', 'gmm.')
                    policy_state_dict[new_key] = value
                
            # Load weights into the model
            policy.load_state_dict(policy_state_dict, strict=True)
            
            # Reset LSTM hidden states for new_episode
            policy.reset_hidden_states(batch_size=1)
            
            print("Successfully loaded IsaacLab BC policy")
            print(f"Loaded {len(policy_state_dict)} weight tensors")

            # Log the structure of self.lstm and self.gmm
            print("=" * 60)
            print("MODEL STRUCTURE LOGGING")
            print("=" * 60)

            print("\n🔍 LSTM Module Structure:")
            print("-" * 40)
            for name, param in policy.lstm.named_parameters():
                print(f"Key: {name}")
                print(f"Shape: {param.shape}")
                print("-" * 20)

            print("\n🔍 GMM Module Structure:")
            print("-" * 40)
            for name, param in policy.gmm.named_parameters():
                print(f"Key: {name}")
                print(f"Shape: {param.shape}")
                print("-" * 20)

            print("=" * 60)
            print("END MODEL STRUCTURE LOGGING")
            print("=" * 60)
            
            return policy
            
        except Exception as e:
            self.get_logger().error(f"Error loading policy: {e}")
            raise

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

    def check_keyboard_input(self):
        """Check for keyboard input and handle commands"""
        try:
            key = self.keyboard.get_key()
            if key is None:
                return
                
            key = key.lower()
            
            if key == ' ':  # Space bar - start/resume
                if not self.is_running:
                    self.start_policy()
                    
            elif key == 's':  # S - stop
                if self.is_running:
                    self.stop_policy()
                    
            elif key == 'r':  # R - reset episode
                self.reset_episode()
                
            elif key == 'z':  # Z - toggle zero vector mode
                self.toggle_zero_vector_mode()
                
            elif key == 'q':  # Q - quit (remove ord(key) == 3 check)
                self.shutdown_requested = True
                print("Shutdown requested...")
                raise KeyboardInterrupt("User requested shutdown")
                
        except KeyboardInterrupt:
            # Re-raise KeyboardInterrupt to allow proper handling
            raise
        except Exception as e:
            print(f"Keyboard input error: {e}")

    def start_policy(self):
        """Start policy execution"""
        print("Starting BC policy execution...")
        self.update_status("Status: RUNNING - Press S to stop, R to reset, Q to quit")
        self.is_running = True
        self.episode_active = True
        # Reset LSTM hidden states for new episode
        self.policy.reset_hidden_states(batch_size=1)
    
    def stop_policy(self):
        """Stop policy execution"""
        print("Stopping BC policy execution...")
        self.update_status("Status: STOPPED - Press SPACE to start, R to reset, Q to quit")
        self.is_running = False
        self.episode_active = False
    
    def reset_episode(self):
        """Reset for new episode"""
        self.policy.reset_hidden_states(batch_size=1)
        self.observation_buffer.clear()  # Clear sequence buffer
        # Reset gripper state
        self.object_grasped = False
        self.gripper_goal_state = 'unknown'
        print("Episode reset - LSTM hidden states cleared and gripper state reset")
        status = "RUNNING" if self.is_running else "STOPPED"
        action = "S to stop" if self.is_running else "SPACE to start"
        self.update_status(f"Status: {status} (RESET) - {action}, R to reset, Q to quit")
    
    def toggle_zero_vector_mode(self):
        """Toggle zero vector test mode"""
        self.zero_vector_mode = not self.zero_vector_mode
        mode_status = "ENABLED" if self.zero_vector_mode else "DISABLED"
        print(f"Zero Vector Mode: {mode_status}")
        
        if self.zero_vector_mode:
            self.update_status(f"Status: ZERO VECTOR MODE - Using 48D zeros for inference")
        else:
            status = "RUNNING" if self.is_running else "STOPPED"
            action = "S to stop" if self.is_running else "SPACE to start"
            self.update_status(f"Status: {status} - {action}, Z for zero mode, Q to quit")
    
    # Create observation dictionary for the policy x_t -> Input to the policy
    def create_observation(self) -> Optional[Dict[str, torch.Tensor]]:
        """Create observation dictionary from current robot state"""
        # Extract end-effector position
        eef_pos = np.array([
            self.current_eef_pose.pose.position.x,
            self.current_eef_pose.pose.position.y,
            self.current_eef_pose.pose.position.z
        ])
        
        # Extract end-effector quaternion from ROS message [qx, qy, qz, qw]
        eef_quat_ros = np.array([
            self.current_eef_pose.pose.orientation.x,  # qx
            self.current_eef_pose.pose.orientation.y,  # qy  
            self.current_eef_pose.pose.orientation.z,  # qz
            self.current_eef_pose.pose.orientation.w   # qw
        ])
        
        # TRANSFORM: Convert from ROS [qx, qy, qz, qw] to IsaacLab [qw, qx, qy, qz] for policy input
        eef_quat_sim = np.array([
            eef_quat_ros[3],  # qw
            eef_quat_ros[0],  # qx
            eef_quat_ros[1],  # qy
            eef_quat_ros[2]   # qz
        ])
        
        # Extract gripper positions 
        gripper_pos = self.current_gripper_positions
        
        # Compute object observations
        object_state = self.compute_object_observations()
        
        # Return dictionary with proper keys and tensor format (using IsaacLab format for policy)
        obs_dict = {
            'eef_pos': torch.from_numpy(eef_pos).float().unsqueeze(0).to(self.device),
            'eef_quat': torch.from_numpy(eef_quat_sim).float().unsqueeze(0).to(self.device),
            'gripper_pos': torch.from_numpy(gripper_pos).float().unsqueeze(0).to(self.device),
            'object': torch.from_numpy(object_state).float().unsqueeze(0).to(self.device)
        }
        
        return obs_dict
    
    def create_zero_observation(self) -> Dict[str, torch.Tensor]:
        """Create observation dictionary with all zeros (48D total)"""
        obs_dict = {
            'eef_pos': torch.zeros(1, self.seq_length, 3, device=self.device),      # [1, seq_length, 3]
            'eef_quat': torch.zeros(1, self.seq_length, 4, device=self.device),     # [1, seq_length, 4]  
            'gripper_pos': torch.zeros(1, self.seq_length, 2, device=self.device),  # [1, seq_length, 2]
            'object': torch.zeros(1, self.seq_length, 39, device=self.device)       # [1, seq_length, 39]
        }
        
        return obs_dict
    
    def control_loop(self):
        """Main control loop - runs at specified frequency - gets called by the timer"""
        if not self.is_running or not self.episode_active:
            return
            
        try:
            if self.zero_vector_mode:
                # Use zero observation directly
                seq_obs_dict = self.create_zero_observation()
            else:
                # Create current observation x_t
                obs_dict = self.create_observation()
                
                # Add to sequence buffer
                self.observation_buffer.append(obs_dict)
                
                # Maintain buffer length
                if len(self.observation_buffer) > self.seq_length:
                    # Remove oldest observation if buffer exceeds sequence length
                    self.observation_buffer.pop(0)
                
                # Pad buffer if needed (for start of episode)
                while len(self.observation_buffer) < self.seq_length:
                    self.observation_buffer.insert(0, obs_dict)  # Repeat first observation
                
                # Create sequence tensor
                seq_obs_dict = self.create_sequence_observation()
        
            # Run policy inference
            with torch.no_grad():
                action = self.policy(seq_obs_dict, deterministic=self.deterministic)
                
            # Convert action to numpy array for processing
            action_np = action.cpu().numpy().squeeze()
            
            # Log action output when in zero vector mode
            if self.zero_vector_mode:
                self.get_logger().info(f"Zero vector input -> Action output: {action_np}", throttle_duration_sec=1.0)
    
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
                self.get_logger().warn("Invalid quaternion received, skipping control step")
                return
            
            # In zero vector mode, don't send commands to robot - just log the outputs
            if self.zero_vector_mode:
                self.get_logger().info(f"Zero mode - Position: [{position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}]", throttle_duration_sec=1.0)
                self.get_logger().info(f"Zero mode - Quaternion: [{quaternion_ros[0]:.4f}, {quaternion_ros[1]:.4f}, {quaternion_ros[2]:.4f}, {quaternion_ros[3]:.4f}]", throttle_duration_sec=1.0)
                self.get_logger().info(f"Zero mode - Gripper: {gripper_command:.4f}", throttle_duration_sec=1.0)
                return
            
            # --- Gripper Control Logic ---
            desired_gripper_state = 'closed' if gripper_command < 0 else 'open'
            
            # Execute gripper action if the state has changed
            if desired_gripper_state != self.gripper_goal_state:
                if desired_gripper_state == 'open' and not self.object_grasped:
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
    
        except Exception as e:
            self.get_logger().error(f"Error in control loop: {e}")
            self.is_running = False

    def create_sequence_observation(self) -> Dict[str, torch.Tensor]:
        """Create sequence observation from buffer"""
        seq_obs = {}
        
        # Stack observations across time dimension
        for key in ['eef_pos', 'eef_quat', 'gripper_pos', 'object']:
            obs_list = [obs[key] for obs in self.observation_buffer]
            # Stack to create [batch_size=1, seq_length, feature_dim]
            seq_obs[key] = torch.stack(obs_list, dim=1)
        
        return seq_obs
        
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

    def publish_observation_debug(self, obs_dict: Dict[str, torch.Tensor]):
        """Publish observation for debugging purposes"""
        try:
            # Concatenate all observation components into a single array (same as policy input)
            obs_tensor = torch.cat([
                obs_dict['eef_pos'],      # [1, 3]
                obs_dict['eef_quat'],     # [1, 4]  
                obs_dict['gripper_pos'],  # [1, 2]
                obs_dict['object']        # [1, 39]
            ], dim=-1)  # Result: [1, 48]
            
            # Convert to numpy and flatten
            obs_np = obs_tensor.cpu().numpy().flatten()  # [48]
            
            # Create and publish message
            obs_msg = Float64MultiArray()
            obs_msg.data = obs_np.tolist()
            self.observation_pub.publish(obs_msg)
                
        except Exception as e:
            self.get_logger().warn(f"Error publishing observation debug: {e}")
        

class KeyboardInput:
    """Handles keyboard input in a non-blocking way"""
    
    def __init__(self):
        try:
            self.old_settings = termios.tcgetattr(sys.stdin)
            self.terminal_setup = False
        except:
            # Handle case where stdin is not a terminal
            self.old_settings = None
            self.terminal_setup = False
        
    def setup_terminal(self):
        """Setup terminal for non-blocking input"""
        if not self.terminal_setup and self.old_settings is not None:
            try:
                tty.setraw(sys.stdin.fileno())
                self.terminal_setup = True
            except:
                # Handle case where terminal setup fails
                pass
        
    def restore_terminal(self):
        """Restore terminal settings"""
        if self.terminal_setup and self.old_settings is not None:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
                self.terminal_setup = False
            except:
                # Handle case where terminal restore fails
                pass
        
    def get_key(self):
        """Get a single keypress if available"""
        if not self.terminal_setup:
            self.setup_terminal()
            
        try:
            if select.select([sys.stdin], [], [], 0.0)[0]:
                return sys.stdin.read(1)
        except:
            # Handle case where stdin reading fails
            pass
        return None

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="BC Policy Runner for Franka Robot")
    parser.add_argument("--policy", type=str, required=True,
                       help="Path to the trained BC policy file (.pt)")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to run inference on (cpu or cuda)")
    parser.add_argument("--deterministic", action="store_true",
                       help="Use deterministic policy inference")
    parser.add_argument("--frequency", type=float, default=20.0,
                       help="Control frequency in Hz")
    
    args = parser.parse_args()
    
    # Initialize ROS2
    rclpy.init()
    
    node = None
    try:
        node = BCPolicyRunner(
            policy_path=args.policy,
            device=args.device,
            deterministic=args.deterministic,
            control_frequency=args.frequency
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