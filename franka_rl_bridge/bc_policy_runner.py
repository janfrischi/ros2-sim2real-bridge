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
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import tf2_ros

from scipy.spatial.transform import Rotation as R

# Define the RNNGMMActorNetwork class
class RNNGMMActorNetwork(nn.Module):
    """LSTM + GMM Actor Network for Behavior Cloning - IsaacLab Compatible"""
    
    def __init__(self, obs_dim: int = 48, action_dim: int = 8, hidden_dim: int = 400, 
                 num_layers: int = 2, num_modes: int = 5, min_std: float = 0.0001,
                 std_activation: str = "softplus", low_noise_eval: bool = True):
        super().__init__()
        
        self.obs_dim = obs_dim # Dimension of the observation space
        self.action_dim = action_dim # Dimension of the action space "Absolute end-effector pose + gripper command"
        self.hidden_dim = hidden_dim # Dimension of LSTM hidden state
        self.num_layers = num_layers # Number of LSTM layers
        self.num_modes = num_modes # Number of Gaussians in GMM (same as num_nodes in IsaacLab)
        self.min_std = min_std
        self.std_activation = std_activation
        self.low_noise_eval = low_noise_eval
        
        # LSTM backbone - Processes sequential observations
        self.lstm = nn.LSTM(
            input_size=obs_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Gaussian Mixture Model (GMM) heads
        # Mean, scale (std), and logits for mixing weights
        self.mean_head = nn.Linear(hidden_dim, num_modes * action_dim)    # 400 -> 40
        self.scale_head = nn.Linear(hidden_dim, num_modes * action_dim)   # 400 -> 40  
        self.logits_head = nn.Linear(hidden_dim, num_modes)               # 400 -> 5
        
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
    def forward(self, obs: Dict[str, torch.Tensor], deterministic: bool = False) -> torch.Tensor:
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
            obs['eef_pos'],      # [batch_size, 3]
            obs['eef_quat'],     # [batch_size, 4]  
            obs['gripper_pos'],  # [batch_size, 2]
            obs['object']        # [batch_size, 39]
        ], dim=-1)  # Result: [batch_size, 48]
        
        # Handle single step input
        if obs_tensor.dim() == 2:
            obs_tensor = obs_tensor.unsqueeze(1)  # Add sequence dimension
            
        batch_size, seq_len, _ = obs_tensor.shape
        
        # Initialize hidden states if needed
        if self.hidden_states is None:
            self.reset_hidden_states(batch_size)
            
        # LSTM forward pass
        # lstm_out contains the processed features that will be fed into the GMM heads
        # hidden_states is a tuple (h_n, c_n) containing the hidden and cell states
        lstm_out, self.hidden_states = self.lstm(obs_tensor, self.hidden_states)
        
        # Take the last timestep output
        lstm_out = lstm_out[:, -1, :]  # [batch_size, hidden_dim]
        
        # GMM parameters with IsaacLab-compatible processing - the .view() operation organizes the flat output into the desired shape
        means = self.mean_head(lstm_out).view(batch_size, self.num_modes, self.action_dim) # Reshape to [batch_size, num_modes, action_dim]
        
        # Apply softplus activation and clamping
        scales = torch.nn.functional.softplus(self.scale_head(lstm_out)).view(batch_size, self.num_modes, self.action_dim) # Reshape to [batch_size, num_modes, action_dim]
        scales = torch.clamp(scales, min=self.min_std)
        
        # Apply low noise evaluation if enabled
        if self.low_noise_eval and not self.training:
            scales = scales * 0.1  # Reduce noise during evaluation
            
        logits = self.logits_head(lstm_out)  # [batch_size, num_modes]
        
        # Handle deterministic vs stochastic action selection from the Gaussian Mixture model
        if deterministic:
            # Return mean of most likely mode
            # Determine the mode probalities and select the best mode
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
            # Create the Gaussian Distribution
            gaussian_dist = torch.distributions.Normal(selected_means, selected_scales)
            # Sample actions from the Gaussian distribution
            actions = gaussian_dist.sample()
            return actions


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
        
        # Zero vector test mode
        self.zero_vector_mode = False
        
        # Object state storage - hardcoded for now
        self.cube_positions = {
            'cube_1': np.array([0.5, 0.2, 0.0203]),
            'cube_2': np.array([0.5, 0.4, 0.0203]),
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
            qos_profile
        )
        
        self.gripper_state_sub = self.create_subscription(
            JointState,
            '/fr3_gripper/joint_states',
            self.gripper_state_callback,
            qos_profile
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
        
        # Add observation publisher for debugging
        self.observation_pub = self.create_publisher(
            Float64MultiArray,
            '/bc_policy/observations',
            qos_profile
        )
        
        # TF2 setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Control timer - control loop runs at specified frequency
        self.control_timer = self.create_timer(
            1.0 / self.control_frequency,
            self.control_loop
        )
        
        # Keyboard input timer (check for keypress every 50ms)
        self.keyboard_timer = self.create_timer(0.05, self.check_keyboard_input)
        
        self.print_instructions()

    def eef_pose_callback(self, msg: PoseStamped):
        """Callback for end-effector pose updates"""
        self.current_eef_pose = msg # Quaternion is in x, y, z, w format
    
    def gripper_state_callback(self, msg: JointState):
        """Callback for gripper state updates"""
        if len(msg.position) >= 2:
            # Gripper should have symmetric but opposite values: [+value, -value]
            finger_1_pos = msg.position[0]  # First finger (positive)
            finger_2_pos = -msg.position[1] if msg.position[1] > 0 else msg.position[1]  # Second finger (negative)
            
            self.current_gripper_positions = np.array([finger_1_pos, finger_2_pos])
        else:
            # Fallback: create symmetric gripper positions
            if len(msg.position) >= 1:
                pos = msg.position[0]
                self.current_gripper_positions = np.array([pos, -pos])
    
    def print_instructions(self):
        """Print keyboard control instructions"""
        # Clear screen and print instructions with proper formatting
        #print("\033[2J\033[H")  # Clear screen and move cursor to top
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

        #TODO: Change convention to match IsaacLab
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

        # TODO: Check convention because of negative x
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

    def load_policy(self, policy_path: str) -> RNNGMMActorNetwork:
        """Load the trained BC policy from checkpoint"""
        try:
            checkpoint = torch.load(policy_path, map_location=self.device)
            
            # Extract model state dict - handle IsaacLab checkpoint format
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'policy' in checkpoint:
                state_dict = checkpoint['policy']
            elif 'BC_RNN_GMM' in checkpoint:  # IsaacLab format
                state_dict = checkpoint['BC_RNN_GMM']['policy']
            else:
                state_dict = checkpoint
            
            # Create network with IsaacLab-compatible parameters
            policy = RNNGMMActorNetwork(
                obs_dim=48,
                action_dim=8,
                hidden_dim=400,
                num_layers=2,
                num_modes=5,
                min_std=0.0001,  # Match IsaacLab exactly
                std_activation="softplus",
                low_noise_eval=True
            ).to(self.device)
            
            # Load weights
            policy.load_state_dict(state_dict, strict=False)
            policy.reset_hidden_states(batch_size=1)
            
            self.get_logger().info(f"Successfully loaded IsaacLab BC policy from {policy_path}")
            return policy
            
        except Exception as e:
            self.get_logger().error(f"Error loading policy: {e}")
            raise
    
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
                
            elif key == 'q' or ord(key) == 3:  # Q or Ctrl+C - quit
                self.shutdown_requested = True
                self.get_logger().info("Shutdown requested...")
                rclpy.shutdown()
                
        except Exception as e:
            self.get_logger().warn(f"Keyboard input error: {e}")
    
    def start_policy(self):
        """Start policy execution"""
        self.get_logger().info("Starting BC policy execution...")
        self.update_status("Status: RUNNING - Press S to stop, R to reset, Q to quit")
        self.is_running = True
        self.episode_active = True
        # Reset LSTM hidden states for new episode
        self.policy.reset_hidden_states(batch_size=1)
    
    def stop_policy(self):
        """Stop policy execution"""
        self.get_logger().info("Stopping BC policy execution...")
        self.update_status("Status: STOPPED - Press SPACE to start, R to reset, Q to quit")
        self.is_running = False
        self.episode_active = False
    
    def reset_episode(self):
        """Reset for new episode"""
        self.policy.reset_hidden_states(batch_size=1)
        self.get_logger().info("Episode reset - LSTM hidden states cleared")
        status = "RUNNING" if self.is_running else "STOPPED"
        action = "S to stop" if self.is_running else "SPACE to start"
        self.update_status(f"Status: {status} (RESET) - {action}, R to reset, Q to quit")
    
    def toggle_zero_vector_mode(self):
        """Toggle zero vector test mode"""
        self.zero_vector_mode = not self.zero_vector_mode
        mode_status = "ENABLED" if self.zero_vector_mode else "DISABLED"
        self.get_logger().info(f"Zero Vector Mode: {mode_status}")
        
        if self.zero_vector_mode:
            self.update_status(f"Status: ZERO VECTOR MODE - Using 48D zeros for inference")
        else:
            status = "RUNNING" if self.is_running else "STOPPED"
            action = "S to stop" if self.is_running else "SPACE to start"
            self.update_status(f"Status: {status} - {action}, Z for zero mode, Q to quit")
    
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
            'eef_pos': torch.zeros(1, 3, device=self.device),      # [1, 3] - zeros
            'eef_quat': torch.zeros(1, 4, device=self.device),     # [1, 4] - zeros  
            'gripper_pos': torch.zeros(1, 2, device=self.device),  # [1, 2] - zeros
            'object': torch.zeros(1, 39, device=self.device)       # [1, 39] - zeros
        }
        
        return obs_dict
    
    def control_loop(self):
        """Main control loop - runs at specified frequency"""
        if not self.is_running or not self.episode_active:
            return
        
        # Create observation dictionary - choose between real observations or zero vector
        if self.zero_vector_mode:
            obs_dict = self.create_zero_observation()
            self.get_logger().info("Using 48D zero vector for inference", throttle_duration_sec=2.0)
        else:
            obs_dict = self.create_observation()
            if obs_dict is None:
                self.get_logger().warn("Observation not ready, skipping control step")
                return
        
        # Publish observation to /bc_policy/observations topic for debugging
        self.publish_observation_debug(obs_dict)
        
        try:
            # Run policy inference with dictionary input
            with torch.no_grad():
                action = self.policy(obs_dict, deterministic=self.deterministic)
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
            
            # TRANSFORM: Convert from IsaacLab [qw, qx, qy, qz] to ROS [qx, qy, qz, qw]
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
            
            # Create cartesian pose command: [x, y, z, qx, qy, qz, qw]
            cartesian_pose = np.concatenate([
                position,         # [x, y, z]
                quaternion_ros    # [qx, qy, qz, qw] - ROS format
            ])
            
            # Publish cartesian pose commands
            pose_msg = Float64MultiArray()
            pose_msg.data = cartesian_pose.tolist()
            self.pose_command_pub.publish(pose_msg)
            
            # TODO: Is the interpet gripper command necessary?
            # Publish gripper commands
            gripper_width = self.interpret_gripper_command(gripper_command)
            gripper_msg = Float64MultiArray()
            gripper_msg.data = [gripper_width]
            self.gripper_command_pub.publish(gripper_msg)
    
        except Exception as e:
            self.get_logger().error(f"Error in control loop: {e}")
            self.is_running = False

    def interpret_gripper_command(self, gripper_command: float) -> float:
        """Interpret gripper command from policy output"""
        if gripper_command < -10:  # Close gripper
            return 0.0
        elif gripper_command > 10:  # Open gripper
            return 0.08  # Max gripper width
        else:
            # Linear interpolation between closed and open
            normalized = (gripper_command + 10) / 20.0  # Map [-10, 10] to [0, 1]
            return normalized * 0.08
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            self.keyboard.restore_terminal()
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
        self.old_settings = termios.tcgetattr(sys.stdin)
        # Don't setup terminal immediately
        self.terminal_setup = False
        
    def setup_terminal(self):
        """Setup terminal for non-blocking input"""
        if not self.terminal_setup:
            tty.setraw(sys.stdin.fileno())
            self.terminal_setup = True
        
    def restore_terminal(self):
        """Restore terminal settings"""
        if self.terminal_setup:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
            self.terminal_setup = False
        
    def get_key(self):
        """Get a single keypress if available"""
        if not self.terminal_setup:
            self.setup_terminal()
            
        if select.select([sys.stdin], [], [], 0.0)[0]:
            return sys.stdin.read(1)
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
    
    rclpy.init()
    
    node = None
    try:
        node = BCPolicyRunner(
            policy_path=args.policy,
            device=args.device,
            deterministic=args.deterministic,
            control_frequency=args.frequency
        )
        
        rclpy.spin(node)
        
    except KeyboardInterrupt:
        print("\nShutting down BC Policy Runner...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if node is not None:
            node.cleanup()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()