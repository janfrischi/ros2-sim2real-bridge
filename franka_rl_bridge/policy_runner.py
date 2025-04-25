#!/usr/bin/env python3
"""
This node interfaces between the joint_state_listener and policy inference.
It subscribes to joint state observations, runs them through the policy,
and publishes resulting actions for the robot to execute.

This version is designed to work with the cartesian_impedance_controller.cpp
which subscribes to the /policy_outputs topic.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String
from sensor_msgs.msg import JointState
import numpy as np
import time
import argparse
from rclpy.callback_groups import ReentrantCallbackGroup
import csv
import os
import torch
from rclpy.action import ActionClient
from franka_msgs.action import Homing, Move, Grasp
from action_msgs.msg import GoalStatus
from rclpy.duration import Duration
from franka_rl_bridge.policy_inference import PolicyLoader

# Define the PolicyRunner node -> Load the policy, receive joint states, run inference, send control commands
class PolicyRunner(Node):
    def configure_logging(self):
        """Configure logging verbosity based on the debug level."""
        if self.debug_level == 0:
            self.get_logger().set_level(rclpy.logging.LoggingSeverity.WARN)
        elif self.debug_level == 1:
            self.get_logger().set_level(rclpy.logging.LoggingSeverity.INFO)
        elif self.debug_level >= 2:
            self.get_logger().set_level(rclpy.logging.LoggingSeverity.DEBUG)

    def __init__(self, policy_path, device="cpu"):
        # Call constructor of Node class from which PolicyRunner inherits, initialize the node with name 'policy_runner'
        super().__init__('policy_runner')

        # Define the path for the CSV log file in the franka_rl_bridge package
        logs_dir = "/home/pdzuser/franka_ros2_ws/src/franka_rl_bridge/franka_rl_bridge"
        self.log_file_path = os.path.join(logs_dir, "policy_logs.csv")
        
        # Open the CSV file for writing
        self.log_file = open(self.log_file_path, mode='w', newline='')
        self.csv_writer = csv.writer(self.log_file)
        
        # Write CSV headers
        self.csv_writer.writerow([
            "timestamp", 
            "joint_positions", 
            "joint_velocities", 
            "object_position", 
            "target_position", 
            "last_action",
            "interpered_actions"  # <-- Add this column
        ])
        
        self.get_logger().info(f"Logging policy observations to {self.log_file_path}")
        
        # Create parameter for verbosity level
        self.declare_parameter('debug_level', 1)  # 0=minimal, 1=normal, 2=detailed
        self.debug_level = self.get_parameter('debug_level').value

        # Configure logging based on debug level
        self.configure_logging()
        
        # Initialize policy, PolicyLoader is a custom class that loads the policy
        self.get_logger().info(f"Loading policy from {policy_path}")
        self.policy_loader = PolicyLoader(policy_path, device)
        
        # Define target object position (this could be made configurable)
        self.target_position = np.array([0.65, 0.2, 0.5, 1.0, 0.0, 0.0, 0.0])  # x,y,z,qw,qx,qy,qz
        
        # Define object position (this could come from a vision system)
        self.object_position = np.array([0.5, 0.4, 0.055])  # x,y,z on table
        self.object_grasped = False # Flag to indicate if the object is currently grasped

        # Initialize joint positions and velocities
        self.joint_positions = np.array([0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741, 0.0, 0.0])
        self.default_joints = np.array([0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741, 0.0, 0.0])
        self.joint_velocities = np.array([0.0, 0.0, 0.0, 0.0, 0.0 , 0.0, 0.0, 0.0, 0.0])
        self.joint_names = None
        self.ee_pos = np.array([0.0, 0.0, 0.0])  # End-effector position
        
        # Store last action
        self.last_action = torch.zeros((1, self.policy_loader.action_dim), device=self.policy_loader.device)

        # Introduce hold position flag and timer
        self.hold_position_active = False
        self.hold_position_end_time = None

        #---Subscription and Publisher Initialization---
        
        # Callback group for allowing concurrent callbacks
        self.callback_group = ReentrantCallbackGroup()
        
        # Create subscription to joint states, velocities, gripper states, and end-effector position
        self.subscription = self.create_subscription(
            Float64MultiArray,
            '/rl/observations',
            self.observation_callback,
            10,
            callback_group=self.callback_group
        )
        
        # Get joint state names from the /joint_states topic
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10, 
            callback_group=self.callback_group
        )
        
        # Create publisher for processed policy outputs -> Cartesian Impedance Controller subscribes to this topic
        self.policy_outputs_publisher = self.create_publisher(
            Float64MultiArray,
            '/policy_outputs',
            10
        )
        
        # Create publisher for policy status
        self.policy_status_publisher = self.create_publisher(String, '/policy_status', 10)
        
        # --- Gripper Control Initialization ---
        self.gripper_goal_state = 'unknown' # 'open', 'closed', 'unknown'
        self.gripper_max_width = 0.08 # Max width for Franka Hand
        self.gripper_speed = 0.05 # Default speed (m/s)
        self.gripper_force = 30.0 # Default grasp force (N)
        self.gripper_epsilon_inner = 0.002 # Tolerance for successful grasp
        self.gripper_epsilon_outer = 0.002

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

        # Initialize running flag and timer
        self.running = False
        self.was_running_previously = False
        self.create_timer(0.1, self.check_state)  # Regular state check
        
        # Initialize print counter
        self.print_counter = 0
        self.print_frequency = 1000  # Print every 10 iterations
        
        self.get_logger().info('Policy Runner initialized')

        # Start 100Hz policy execution timer
        self.policy_timer = self.create_timer(0.01, self.timer_policy_step, callback_group=self.callback_group)
        self.get_logger().info("Policy execution timer started at 100Hz")

        # Keep robot steady during gripper closure
        self.hold_position_timer = self.create_timer(0.01, self.hold_position_step, callback_group=self.callback_group)

    # Helper to wait for action servers, using the .wait_for_server() method
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
        if not self.homing_client.server_is_ready():
            self.get_logger().error("Homing action server not ready.")
            return

        self.get_logger().info("Sending homing goal...")
        goal_msg = Homing.Goal()
        # Send goal async and forget (or handle future if needed)
        self.homing_client.send_goal_async(goal_msg)
        self.gripper_goal_state = 'open' # Assume homing opens the gripper
        self.get_logger().info("Homing goal sent.")

    # -----------------------------------------------------
    
    def joint_state_callback(self, msg):
        """Receive joint state information including names"""
        self.joint_names = msg.name

    # Process observations from the /rl/observations topic
    # Extract joint positions, gripper positions, joint velocities, gripper velocities, and end-effector position
    def observation_callback(self, msg):
        """Process incoming joint state observations"""
            
        # Extract data from the observation /rl/observations topic
        # Structure: [joint_pos(7), gripper_pos(2), joint_vel(7), gripper_vel(2), ee_pos(3)] -> Total 21
        data = np.array(msg.data) 
        if len(data) != 21:
            self.get_logger().warning(f"Received observation data of unexpected length {len(data)}, expected 21. Skipping.")
            return
        
        # Extract the joint positions and velocities, and gripper positions and velocities and end-effector position
        joint_pos_arm = data[0:7]
        gripper_pos = data[7:9] 
        joint_vel_arm = data[9:16]
        gripper_vel = data[16:18]
        ee_pos = data[18:21]
        # Store the latest ee_pos
        self.ee_pos = ee_pos 

        # Current gripper width is the sum of the gripper positions
        gripper_width = np.sum(gripper_pos)
        # The processed gripper_pos that serve as an input to the policy are defined as policy_input_gripper = -(0.08 -gripper_width)/2
        policy_input_gripper = -(0.08 - gripper_width) / 2
        gripper_pos_processed = np.array([policy_input_gripper, policy_input_gripper])
        
        # Concatenate arm joint positions (7) and processed gripper positions (2)
        self.joint_positions = np.concatenate((joint_pos_arm, gripper_pos_processed))
        self.joint_velocities = np.concatenate((joint_vel_arm, gripper_vel))

        # Define default joint positions defined in the rl training
        default_joints = np.array([0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741, 0.0, 0.0])

        # Calculate relative joint positions: joint_positions = q_robot - q_default
        self.joint_positions = self.joint_positions - default_joints
            
    def timer_policy_step(self):
        """Run policy inference and send actions at fixed 100Hz loop."""
        # Run the policy inference if active
        #if self.running:
        self.run_policy_inference()

    def execute_action(self, action_dict):
        """Execute the action returned by the policy "dictionary" with safety checks"""
        # Extract joint positions and gripper command from the action dictionary
        raw_joint_positions = action_dict["joint_positions"]
        # Gripper command is the last element from the policy output, gripper command is a scalar value
        gripper_command = action_dict["gripper_command"]

        # # OVERRIDE GRIPPER COMMAND IF OBJECT IS GRASPED
        # if self.object_grasped:
        #     gripper_command = -2.5  # Ensure it stays closed (below -2 threshold)

        # --- Joint Position Control ---
        # SAFETY CHECK 1: Validate that we have valid joint positions (not NaN or inf)
        if not all(np.isfinite(pos) for pos in raw_joint_positions):
            self.get_logger().error(f"Safety violation: Non-finite joint position detected: {raw_joint_positions}")
            return
        
        # Process joint commands (apply scaling and offsets) -> q_target = 0.5 * policy_output + q_default
        joint_positions = self.process_joint_commands(raw_joint_positions)
        
        # SAFETY CHECK 2: Joint position limits
        # Franka joint limits as defined in the URDF
        joint_limits = [
            (-2.8973, 2.8973),    # panda_joint1
            (-1.7628, 1.7628),    # panda_joint2
            (-2.8973, 2.8973),    # panda_joint3
            (-3.0718, -0.0698),   # panda_joint4
            (-2.8973, 2.8973),    # panda_joint5
            (-0.0175, 3.7525),    # panda_joint6
            (-2.8973, 2.8973)     # panda_joint7
        ]
        
        # Clamp joint positions to their respective limits
        clamped_joint_positions = []
        for i, pos in enumerate(joint_positions):

            lower_limit, upper_limit = joint_limits[i]
            # Check if the position is within limits, if position is below lower limit or above upper limit, clamp it
            # Lower limit check
            if pos < lower_limit:
                #self.get_logger().warning(f"Joint {i+1} position {pos:.4f} below lower limit {lower_limit:.4f}. Clamping to {lower_limit:.4f}.")
                clamped_joint_positions.append(lower_limit)
            # Upper limit check
            elif pos > upper_limit:
                #self.get_logger().warning(f"Joint {i+1} position {pos:.4f} above upper limit {upper_limit:.4f}. Clamping to {upper_limit:.4f}.")
                clamped_joint_positions.append(upper_limit)
            # If within limits, append the position as is
            else:
                clamped_joint_positions.append(pos)

        # Replace the original joint positions with the clamped ones
        joint_positions = clamped_joint_positions
        
        # Publish processed policy outputs to the /policy_outputs topic
        # The processed outputs are a 8D vector: 7 joint positions + 1 gripper command
        try:
            processed_outputs_msg = Float64MultiArray()
            # Explicitly convert all values to Python float to ensure proper type conversion
            processed_data = [float(pos) for pos in joint_positions]
            # Append gripper command as the last element -> The output is an 8D vector, 7 joint positions + 1 gripper command
            processed_data.append(float(gripper_command)) # Keep original gripper command for controller
            
            # Add processed data to message and publish it
            processed_outputs_msg.data = processed_data
            self.policy_outputs_publisher.publish(processed_outputs_msg)

        except Exception as e:
            self.get_logger().error(f"Error publishing processed outputs: {e}")
            return
        # --- End Joint Position Control ---

        # TODO: Double check the following logic
        # --- Gripper Command Execution ---
        self.desired_gripper_state = 'closed' if gripper_command <= -1 else 'open'
        # Log the gripper command
        self.get_logger().info(f"Gripper command: {gripper_command:.3f} -> Desired state: {self.desired_gripper_state}")

        # Only send a command if the desired state is different from the current goal state
        if self.desired_gripper_state != self.gripper_goal_state:
            if self.desired_gripper_state == 'open' and not self.object_grasped:
                self.get_logger().info("Entering gripper open state.")
                if self.move_client.server_is_ready():
                    self.get_logger().info(f"Sending Move goal (open) - Width: {self.gripper_max_width}, Speed: {self.gripper_speed}")
                    # Create a goal message for the Move action
                    goal_msg = Move.Goal()
                    goal_msg.width = self.gripper_max_width
                    goal_msg.speed = self.gripper_speed
                    self.move_client.send_goal_async(goal_msg)
                    # Set the gripper goal state to 'open'
                    self.gripper_goal_state = 'open'
                    # Set the object grasped flag to False
                    self.object_grasped = False
                    self.get_logger().info("Object grasp flag set to False (gripper opening).")
                else:
                    self.get_logger().warning("Move action server not ready. Cannot open gripper.")

            elif self.desired_gripper_state == 'closed':
                self.get_logger().info(f"Sending Grasp goal (close) - Width: 0.0, Speed: {self.gripper_speed}, Force: {self.gripper_force}")
                self.get_logger().info("Pausing policy execution until grasp completes.")
                #self.stop() # Pause policy execution

                # Create a goal message for the Grasp action
                goal_msg = Grasp.Goal()
                goal_msg.width = 0.0 # Target width when grasping (close fully)
                goal_msg.speed = self.gripper_speed
                goal_msg.force = self.gripper_force
                goal_msg.epsilon.inner = self.gripper_epsilon_inner
                goal_msg.epsilon.outer = self.gripper_epsilon_outer

                self.stop() # Pause policy execution

                # Send goal and register callback for the result
                # Once the grasp is complete, we resume policy execution
                grasp_future = self.grasp_client.send_goal_async(goal_msg)
                grasp_future.add_done_callback(self.grasp_goal_response_callback) # Check if goal was accepted

                # # Send current joint positions to hold the robot steady
                # processed_outputs_msg = Float64MultiArray()
                # # Explicitly convert all values to Python float to ensure proper type conversion
                # processed_data = [float(pos) for pos in (self.joint_positions[:7] + self.default_joints[:7])]
                # # Append gripper command as the last element -> The output is an 8D vector, 7 joint positions + 1 gripper command
                # processed_data.append(float(gripper_command)) # Keep original gripper command for controller
                
                # # Add processed data to message and publish it
                # processed_outputs_msg.data = processed_data
                # self.policy_outputs_publisher.publish(processed_outputs_msg)
                self.activate_hold_position(2)
                self.gripper_goal_state = 'closed'
        # --- End Gripper Command Execution ---

    def run_policy_inference(self):
        """Run the loaded policy on current observations"""
        
        # Dynamically update the object position if grasped
        if self.object_grasped:
            self.object_position = self.ee_pos.copy()
            
        # Create observation for policy 
        obs = self.policy_loader.create_observation(
            joint_pos=self.joint_positions,
            joint_vel=self.joint_velocities,
            object_pos=self.object_position,
            target_pos=self.target_position
        )
        
        # Increment print counter
        self.print_counter += 1

        # --- SAFETY: Prevent end-effector from going below table + margin ---
        min_ee_z = 0.03
        current_ee_z = self.ee_pos[2] if self.ee_pos is not None and len(self.ee_pos) > 2 else None
        if current_ee_z is not None and current_ee_z < min_ee_z:
            self.get_logger().warning(f"Safety: End-effector z={current_ee_z:.3f} below safety threshold ({min_ee_z:.3f}). ""Sending safe joint positions to avoid table collision.")
            # Send current joint positions (relative, so add default) as a safe fallback
            default_joints = np.array([0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741])
            safe_joint_positions = (self.joint_positions[:7] + default_joints).tolist()
            # Use last gripper command or open gripper
            safe_gripper_command = float(self.last_action[0, -1].item()) if self.last_action is not None else 1.0
            safe_action_dict = {
                "joint_positions": safe_joint_positions,
                "gripper_command": safe_gripper_command
            }
            self.execute_action(safe_action_dict)
            return
        
        # Log detailed information about the observation every `print_frequency` iterations
        if self.print_counter % self.print_frequency == 0:
            self.get_logger().info("==== Policy Observation Details ====")
            self.get_logger().info(f"Joint positions (9):  [{', '.join([f'{x:.3f}' for x in self.joint_positions])}]")
            self.get_logger().info(f"Joint velocities (9): [{', '.join([f'{x:.3f}' for x in self.joint_velocities])}]")
            self.get_logger().info(f"Object position (3):  [{', '.join([f'{x:.3f}' for x in self.object_position])}]")
            self.get_logger().info(f"Target position (7):  [{', '.join([f'{x:.3f}' for x in self.target_position])}]")
            
            # Try to get the last action if it exists
            if self.last_action is not None:
                last_action_np = self.last_action.detach().cpu().numpy()[0]
                self.get_logger().info(f"Last action (8):      [{', '.join([f'{x:.3f}' for x in last_action_np])}]")
            self.get_logger().info("==================================")
        
        # Run inference at the same frequency as training frequency in simulation
        try:
            if self.running:
                action = self.policy_loader.run_inference(obs)

            else: 
                # If not running, just hold the last action
                action = self.last_action

            # Extract joint positions and gripper command from the action, interpreted action is a dictionary: {"joint_positions": joint_positions, "gripper_command": gripper_command,"gripper_state": gripper_state}
            interpered_actions = self.policy_loader.interpret_action(action)

            # Update the last action
            self.last_action = action.detach().clone()
            
            # Log the interpreted action
            self.get_logger().info(f"Last action: {action}")
            self.get_logger().info(f"Interpreted action: {interpered_actions}")
            
            # --- Compose interpered_actions as a list for logging ---
            interpered_actions_vec = list(interpered_actions["joint_positions"]) + [interpered_actions["gripper_command"]]

            # Log observations and actions to CSV
            timestamp = time.time()
            last_action_np = self.last_action.detach().cpu().numpy()[0] if self.last_action is not None else [None] * 8
            self.csv_writer.writerow([
                timestamp,
                self.joint_positions.tolist(),
                self.joint_velocities.tolist(),
                self.object_position.tolist(),
                self.target_position.tolist(),
                last_action_np,
                interpered_actions_vec  # <-- Log interpered_actions
            ])
            
            # Send control commands to the robot, joint commands are handled in cartesian_impedance_controller, gripper command is handled here
            self.execute_action(interpered_actions)
            
        except Exception as e:
            self.get_logger().error(f"Error running policy inference: {e}")

    def process_joint_commands(self, policy_output):
        """Convert policy outputs to joint position commands for the real robot.
        
        Observations are defined as: obs = q_robot - q_default
        Policy outputs are rescaled with a factor of 0.5.
        Target joint commands are defined as: q_target = 0.5 * policy_output + q_default
        """
        # Get default joint positions for Franka
        default_joints = {
            "panda_joint1": 0.0,
            "panda_joint2": -0.569,
            "panda_joint3": 0.0,
            "panda_joint4": -2.810,
            "panda_joint5": 0.0,
            "panda_joint6": 3.037,
            "panda_joint7": 0.741
        }
        
        # Ensure joint_positions are available
        if self.joint_positions is None:
            self.get_logger().error("Joint positions are not available for processing.")
            return None
        
        # Extract arm actions (first 7 values) from policy output
        arm_actions = policy_output[:7]
        
        # Rescale policy outputs with a factor of 0.5
        scaling_factor = 0.5
        scaled_actions = [scaling_factor * action for action in arm_actions]
        
        # Calculate target joint commands: q_target = 0.5 * policy_output + q_default
        joint_positions = [
            scaled_action + q_default
            for scaled_action, q_default in zip(scaled_actions, default_joints.values())
        ]
        
        # # Debug logging
        # if self.print_counter % self.print_frequency == 0:
        #     self.get_logger().info("==== Policy Action Processing ====")
        #     self.get_logger().info(f"Observations (q_robot - q_default): [{', '.join([f'{x:.3f}' for x in self.joint_positions])}]")
        #     self.get_logger().info(f"Raw policy outputs:  [{', '.join([f'{x:.3f}' for x in arm_actions])}]")
        #     self.get_logger().info(f"Scaled policy outputs (x{scaling_factor}): [{', '.join([f'{x:.3f}' for x in scaled_actions])}]")
        #     self.get_logger().info(f"Target joint positions (q_target): [{', '.join([f'{x:.3f}' for x in joint_positions])}]")
        #     self.get_logger().info("================================")
        
        return joint_positions
    
    def send_hold_position_command(self):
        """Publish current joint positions to hold the robot steady."""
        # Add back q_default offset to the observed joint positions
        q_default = np.array([0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741])
        hold_joint_positions = (self.joint_positions[:7] + q_default).tolist()
        gripper_command = self.last_action[0, -1].item() if self.last_action is not None else 1.0

        processed_outputs_msg = Float64MultiArray()
        processed_data = [float(pos) for pos in hold_joint_positions]
        processed_data.append(float(gripper_command))  # Keep the last gripper command
        processed_outputs_msg.data = processed_data 

        self.policy_outputs_publisher.publish(processed_outputs_msg)
        #self.get_logger().info("Published hold position to maintain arm pose during gripper closure.")

    def activate_hold_position(self, duration_sec):
        """Activate hold position for a specified duration."""
        self.hold_position_active = True
        self.hold_position_end_time = self.get_clock().now() + Duration(seconds=duration_sec)
        #self.get_logger().info(f"Hold position activated for {duration_sec} seconds.")

    def hold_position_step(self):
        """Periodically send hold position commands while in hold mode."""
        if not self.hold_position_active:
            return

        if self.get_clock().now() >= self.hold_position_end_time:
            self.hold_position_active = False
            self.get_logger().info("Hold position mode ended.")
            return

        # Send the hold position command
        self.send_hold_position_command()

    def grasp_goal_response_callback(self, future):
        """Callback when grasp goal response is received (accepted/rejected)."""
        # The future object contains the result of the goal request
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Grasp goal rejected.')
            self.start() # Resume policy even if rejected
            return

        self.get_logger().info('Grasp goal accepted. Waiting for result...')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.grasp_result_callback) # Add callback for the final result

    # Callback when the grasp action completes
    def grasp_result_callback(self, future):
        """Callback triggered when the grasp action completes."""
        result = future.result().result # object result: success, error
        status = future.result().status # status of the action: SUCCEEDED, ABORTED, CANCELED, etc.

        if status == GoalStatus.STATUS_SUCCEEDED:
            if result and result.success:
                self.get_logger().info("Grasp successful. Object is now attached to end-effector.")
                self.object_grasped = True # Set the flag
                self.hold_position_active = False # Deactivate hold position
                # Resume policy execution only if grasp is successful
                self.get_logger().info("Resuming policy execution.")
                self.start()
            else:
                 self.get_logger().warning(f"Grasp action status SUCCEEDED, but result indicates failure: {result.error if result else 'No result object'}")
                 self.object_grasped = False # Ensure flag is false on failure
        elif status == GoalStatus.STATUS_ABORTED:
            self.get_logger().error("Grasp action aborted.")
            self.object_grasped = False # Ensure flag is false on failure
        elif status == GoalStatus.STATUS_CANCELED:
            self.get_logger().warning("Grasp action canceled.")
            self.object_grasped = False # Ensure flag is false on failure
        else:
            self.get_logger().error(f"Grasp action failed with status: {status}")
            self.object_grasped = False # Ensure flag is false on failure

    def check_state(self):
        """Regular check of the node's state"""
        # Add a state tracking variable to the class if it doesn't exist yet
        if not hasattr(self, 'was_running_previously'):
            self.was_running_previously = self.running
        
        # Only log the message if there's a state change from running to not running
        if not self.running and self.was_running_previously:
            self.get_logger().info("Policy runner is paused. Press 's' to start.")
        
        # Update the previous state
        self.was_running_previously = self.running
        
    def start(self):
        """Resume policy execution."""
        self.running = True
        # Publish status update
        status_msg = String()
        status_msg.data = "resumed"
        self.policy_status_publisher.publish(status_msg)
        self.get_logger().info("Policy execution resumed.")
        
    def stop(self):
        """Pause policy execution."""
        self.running = False
        # Publish status update
        status_msg = String()
        status_msg.data = "paused"
        self.policy_status_publisher.publish(status_msg)
        self.get_logger().info("Policy execution paused.")

    def reset_to_home(self):
        """Reset the robot to home position and open gripper."""
        # Define home position joint angles
        home_position = [0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741]
        
        # Stop policy execution temporarily
        self.stop()  # Ensure the robot is paused
        self.object_grasped = False # Ensure object is considered not grasped on reset
        # Reset the last action
        self.last_action = torch.zeros((1, self.policy_loader.action_dim), device=self.policy_loader.device)
        # Reset the object position 
        self.object_position = np.array([0.5, 0.3, 0.055])  # Reset to initial position
        # Reset the gripper state
        self.desired_gripper_state = 'open'  # Reset desired gripper state

        # Send robot to home position
        self.get_logger().info("Resetting robot to home position and opening gripper")
        
        # Publish directly to policy outputs to reset joints
        processed_outputs_msg = Float64MultiArray()
        processed_data = [float(pos) for pos in home_position]
        processed_data.append(-1.0)  # Append a negative value for gripper command to indicate open
        processed_outputs_msg.data = processed_data

        self.policy_outputs_publisher.publish(processed_outputs_msg)

        # Send command to open gripper via action client
        if self.move_client.server_is_ready():
            goal_msg = Move.Goal()
            goal_msg.width = self.gripper_max_width
            goal_msg.speed = self.gripper_speed
            self.move_client.send_goal_async(goal_msg)
            self.gripper_goal_state = 'open' # Update state
        else:
             self.get_logger().warning("Move action server not ready during reset. Cannot open gripper.")

        # Wait a bit for movement to potentially start
        time.sleep(1.0)
        
        # Log that the robot is now in pause mode
        self.get_logger().info("Robot reset command sent. Policy execution is paused.")

    def destroy_node(self):
        """Clean up resources when the node is destroyed"""
        self.log_file.close()
        self.get_logger().info(f"Policy logs saved to {self.log_file_path}")
        # Destroy action clients
        self.homing_client.destroy()
        self.move_client.destroy()
        self.grasp_client.destroy()
        self.get_logger().info("Gripper action clients destroyed.")
        super().destroy_node()


def main(args=None):
    parser = argparse.ArgumentParser(description="Run a trained policy with live robot observations")
    parser.add_argument("--policy", type=str, required=True, help="Path to the trained policy file (.pt or .jit)")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run inference on (cpu or cuda)")
    parser.add_argument("--auto-start", action="store_true", help="Start policy execution immediately")
    
    # Parse ROS args and then our custom args
    filtered_args = rclpy.utilities.remove_ros_args(args=args)
    parsed_args = parser.parse_args(args=filtered_args[1:])
    
    rclpy.init(args=args)
    
    runner = PolicyRunner(parsed_args.policy, parsed_args.device)
    
    # Auto-start if flag is set
    if parsed_args.auto_start:
        runner.start()
    
    # Print instructions
    print("\n" + "="*50)
    print("POLICY RUNNER CONTROLS")
    print("="*50)
    print("Press 's' to START policy execution")
    print("Press 'p' to PAUSE policy execution")
    print("Press 'r' to RESET robot")
    print("Press 'q' to QUIT")
    print("="*50 + "\n")
    
    # Setup keyboard monitoring
    import sys
    import select
    import tty
    import termios
    
    def is_data():
        return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])
    
    # Save terminal settings
    old_settings = termios.tcgetattr(sys.stdin)
    
    try:
        # Set terminal to raw mode
        tty.setcbreak(sys.stdin.fileno())
        
        while rclpy.ok():
            # Process ROS callbacks
            rclpy.spin_once(runner, timeout_sec=0.1)
            
            # Check for keyboard input
            if is_data():
                c = sys.stdin.read(1)
                if c == 's':
                    runner.start()
                    print("\nPolicy execution STARTED")
                elif c == 'p':
                    runner.stop()
                    print("\nPolicy execution PAUSED")
                elif c == 'r':  # New reset command
                    runner.reset_to_home()  
                    print("\nResetting robot to home position")
                elif c == 'q':
                    print("\nExiting...")
                    break
    
    finally:
        # Restore terminal settings
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        runner.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()