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
from geometry_msgs.msg import PoseStamped  # Add PoseStamped import
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
    def __init__(self, policy_path, device="cpu"):
        # Call constructor of Node class from which PolicyRunner inherits, initialize the node with name 'policy_runner'
        super().__init__('policy_runner')

        # Define the path for the CSV log file in the franka_rl_bridge package
        logs_dir = "/home/chris/franka_ros2_ws/src/franka_rl_bridge/franka_rl_bridge"
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
            "object_orientation", 
            "target_position", 
            "last_action",
            "interpreted_actions"
        ])
        
        self.get_logger().info(f"Logging policy observations to {self.log_file_path}")
        
        # Initialize policy, PolicyLoader is a custom class that loads the policy
        self.get_logger().info(f"Loading policy from {policy_path}")
        self.policy_loader = PolicyLoader(policy_path, device)
        
        # Define target object position (this could be made configurable)
        self.target_position = np.array([0.65, 0.2, 0.5, 1.0, 0.0, 0.0, 0.0])  # x,y,z,qw,qx,qy,qz
        
        # Define object position (this could come from a vision system)
        self.object_position = np.array([0.4, -0.2, 0.055])  # x,y,z on table - default value if no perception data
        self.object_orientation = np.array([1.0, 0.0, 0.0, 0.0]) # Default orientation (identity quaternion)
        self.object_grasped = False # Flag to indicate if the object is currently grasped
        self.object_position_received = False  # Flag to track if we've received object position data

        # Initialize joint positions and velocities
        self.joint_positions = np.array([0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741, 0.0, 0.0])
        self.default_joints = np.array([0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741, 0.0, 0.0])
        self.joint_velocities = np.array([0.0, 0.0, 0.0, 0.0, 0.0 , 0.0, 0.0, 0.0, 0.0])
        self.joint_names = None
        self.ee_pos = np.array([0.0, 0.0, 0.0])  # End-effector position
        
        # Initialize the last action to zero
        self.last_action = torch.zeros((1, self.policy_loader.action_dim), device=self.policy_loader.device)

        # Introduce hold position flag and timer
        self.hold_position_active = False
        self.hold_position_end_time = None
        self.hold_position = None

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
        
        # Create subscription to object pose from perception system
        self.object_pose_subscription = self.create_subscription(
            PoseStamped,
            '/perception/object_pose',
            self.object_pose_callback,
            10,
            callback_group=self.callback_group
        )
        self.get_logger().info("Subscribed to /perception/object_pose topic")
           
        # Create publisher for processed policy outputs -> Cartesian Impedance Controller subscribes to this topic
        self.policy_outputs_publisher = self.create_publisher(
            Float64MultiArray,
            '/policy_outputs',
            10
        )
        
        # Create publisher for policy status
        self.policy_status_publisher = self.create_publisher(String, '/policy_status', 10)
        self.get_logger().info("Created publisher for /policy_status topic")
        
        # --- Gripper Control Initialization ---
        self.gripper_goal_state = 'unknown' # 'open', 'closed', 'unknown'
        self.gripper_max_width = 0.08 # Max width for Franka Hand
        self.gripper_speed = 0.05 # Default speed (m/s)
        self.gripper_force = 30.0 # Default grasp force (N)
        self.gripper_epsilon_inner = 0.02 # Tolerance for successful grasp
        self.gripper_epsilon_outer = 0.05

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
        self.was_holding_previously = False
        self.create_timer(0.1, self.check_state)  # Regular state check
        
        # Initialize print counter
        self.print_counter = 0
        self.print_frequency = 50
        
        self.get_logger().info('Policy Runner initialized')

        # Start 100Hz policy execution timer -> Callback function will be called at 100Hz
        self.policy_timer = self.create_timer(0.01, self.run_policy_inference, callback_group=self.callback_group)
        self.get_logger().info("Policy execution timer started at 100Hz")

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
        self.get_logger().info("Sending homing goal...")
        goal_msg = Homing.Goal()
        # Send goal async and forget (or handle future if needed)
        self.homing_client.send_goal_async(goal_msg)
        self.gripper_goal_state = 'open' # Assume homing opens the gripper
        self.get_logger().info("Homing goal sent.")

    # -----------------------------------------------------
    # Process observations from the /rl/observations topic and return the relative joint positions
    def observation_callback(self, msg):
        """Process incoming joint state observations"""
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

        # Gripper width is the sum of the two gripper positions
        gripper_width = np.sum(gripper_pos)
        # The processed gripper_pos that serve as an input to the policy are defined as policy_input_gripper = -(0.08 -gripper_width)/2
        policy_input_gripper = -(0.08 - gripper_width) / 2
        gripper_pos_processed = np.array([policy_input_gripper, policy_input_gripper])
        
        # Concatenate arm joint positions (7) and processed gripper positions (2)
        # Return the relative joint positions
        self.joint_positions = np.concatenate((joint_pos_arm, gripper_pos_processed)) - self.default_joints
        self.joint_velocities = np.concatenate((joint_vel_arm, gripper_vel))

    # Callback for object pose from perception system
    def object_pose_callback(self, msg):
        """Process incoming object pose from perception system"""
        # Extract position from the message
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z
        
        # Extract orientation (quaternion) from the message
        qw = msg.pose.orientation.w
        qx = msg.pose.orientation.x
        qy = msg.pose.orientation.y
        qz = msg.pose.orientation.z
        
        # Update object position and orientation
        self.object_position = np.array([x, y, z])
        self.object_orientation = np.array([qw, qx, qy, qz])
        
        # Set flag to indicate we've received object position data
        self.object_position_received = True
        
        # Debug output (occasionally)
        if hasattr(self, 'print_counter') and self.print_counter % self.print_frequency == 0:
            self.get_logger().info(f"Received object position: [{x:.3f}, {y:.3f}, {z:.3f}]")

    def process_joint_commands(self, policy_output):
        """Convert policy outputs to joint position commands for the real robot.
        
        Observations are defined as: obs = q_robot - q_default
        Policy outputs are rescaled with a factor of 0.5.
        Target joint commands are defined as: q_target = 0.5 * policy_output + q_default
        """
        # Ensure joint_positions are available
        if self.joint_positions is None:
            self.get_logger().error("Joint positions are not available for processing.")
            return None
        
        # Extract arm actions, rescale with 0.5, and add default positions in one step
        scaling_factor = 0.5
        processed_joint_positions = [
            scaling_factor * action + default
            for action, default in zip(policy_output[:7], self.default_joints[:7])
        ]
        
        return processed_joint_positions

    # Process the action returned by the policy
    def execute_action(self, action_dict):
        """Execute the action returned by the policy "dictionary" with safety checks"""
        # Extract joint positions and gripper command from the action dictionary "relative joint positions"
        joint_actions_relative = action_dict["joint_positions"]
        gripper_command = action_dict["gripper_command"]

        # --- Joint Position Control ---
        # SAFETY CHECK 1: Validate that we have valid joint positions (not NaN or inf)
        if not all(np.isfinite(pos) for pos in joint_actions_relative):
            self.get_logger().error(f"Safety violation: Non-finite joint position detected: {joint_actions_relative}")
            return
        
        # Process joint commands (apply scaling and offsets) -> q_target = 0.5 * policy_output + q_default "Absolute joint positions"
        joint_positions_absolute = self.process_joint_commands(joint_actions_relative)
        
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
        for i, pos in enumerate(joint_positions_absolute):
            lower_limit, upper_limit = joint_limits[i]
            if pos < lower_limit:
                clamped_joint_positions.append(lower_limit)
            elif pos > upper_limit:
                clamped_joint_positions.append(upper_limit)
            else:
                clamped_joint_positions.append(pos)

        # Replace the original joint positions with the clamped ones
        joint_positions_absolute = clamped_joint_positions

        # Publish joint position commands to the /policy_outputs topic
        try: 
            processed_policy_outputs_msg = Float64MultiArray()
            processed_data = [float(pos) for pos in joint_positions_absolute]
            processed_data.append(float(gripper_command))
            processed_policy_outputs_msg.data = processed_data
            
            # Publish the processed policy outputs to the /policy_outputs topic
            self.policy_outputs_publisher.publish(processed_policy_outputs_msg)
            
            if self.print_counter % self.print_frequency == 0:  # Only log occasionally
                self.get_logger().info(f"Published policy outputs: {processed_data}")
        except Exception as e:
            self.get_logger().error(f"Error publishing policy outputs: {e}")
            return

    def open_gripper(self):
        """Open the gripper using the action client"""
        goal_msg = Move.Goal()
        goal_msg.width = self.gripper_max_width
        goal_msg.speed = self.gripper_speed
        self.move_client.send_goal_async(goal_msg)
        self.gripper_goal_state = 'open'
        self.object_grasped = False

    def close_gripper(self):
        """Close the gripper using the action client"""
        goal_msg = Grasp.Goal()
        goal_msg.width = self.gripper_max_width
        goal_msg.speed = self.gripper_speed
        goal_msg.force = self.gripper_force
        goal_msg.epsilon.inner = self.gripper_epsilon_inner
        goal_msg.epsilon.outer = self.gripper_epsilon_outer

        # Send the goal and register the callback for the result
        grasp_future = self.grasp_client.send_goal_async(goal_msg)
        grasp_future.add_done_callback(self.grasp_goal_response_callback)

        self.gripper_goal_state = 'closed'

    # Main logic for running policy inference
    def run_policy_inference(self):
        """Run the loaded policy on current observations"""
        
        # Create observation for policy 
        obs = self.policy_loader.create_observation(
            joint_pos=self.joint_positions,
            joint_vel=self.joint_velocities,
            object_pos=self.object_position,
            object_orientation=self.object_orientation,
            target_pos=self.target_position
        )
        
        # Dynamically update the object position if the grasp was successful
        if self.object_grasped:
            self.object_position = self.ee_pos.copy()

        # Increment print counter
        self.print_counter += 1
        
        try:
            # CASE 1: Policy is running normally
            if self.running:
                # Run inference on the observations
                action = self.policy_loader.run_inference(obs)
                if self.object_grasped:
                    action[0, -1] = -4.5
                # Extract joint positions and gripper command from the action
                interpreted_actions = self.policy_loader.interpret_action(action)
                
                # --- Compose interpreted_actions as a list for logging ---
                interpreted_actions_vec = list(interpreted_actions["joint_positions"]) + [interpreted_actions["gripper_command"]]
                
                # Log observations and actions to CSV
                timestamp = time.time()
                last_action_np = action.detach().cpu().numpy()[0]
                self.csv_writer.writerow([
                    timestamp,
                    self.joint_positions.tolist(),
                    self.joint_velocities.tolist(),
                    self.object_position.tolist(),
                    self.object_orientation.tolist(),
                    self.target_position.tolist(),
                    last_action_np,
                    interpreted_actions_vec
                ])

                # Handle gripper command
                gripper_command = action[0, -1]
                desired_gripper_state = 'closed' if gripper_command <= -1 else 'open'
                
                # Execute gripper action if the state has changed
                if desired_gripper_state != self.gripper_goal_state:
                    
                    #--- Gripper Control Logic ---
                    if desired_gripper_state == 'open' and not self.object_grasped:
                        self.open_gripper()
                        return
                    elif desired_gripper_state == 'closed':
                        # Store current joint positions for holding
                        self.hold_position = interpreted_actions
                        # Convert self.joint_positions to format that matches the policy output
                        self.hold_position["joint_positions"] = 2*(self.joint_positions[:7])
                        self.get_logger().info(f"Holding position: {self.hold_position}")
                        # Enter hold mode
                        self.hold_position_active = True
                        self.stop()  # Pause policy execution
                        # Send the grasp command
                        self.close_gripper()
                        self.execute_action(self.hold_position)
                        self.last_action[0, :-1] = torch.tensor(self.joint_positions, device=self.policy_loader.device).unsqueeze(0)
                        self.last_action[0, -1] = -4.5
                        self.get_logger().info(f"Holding position: {self.hold_position}")
                        return
                
                # If gripper action hasn't changed simply execute the action
                self.execute_action(interpreted_actions)
                # Update the last action
                self.last_action = action.detach().clone()
               
            # CASE 2: Policy is in hold position mode (during gripper closure)
            elif self.hold_position_active:
                self.hold_position["gripper_command"] = -4.5
                self.last_action[0, -1] = -4.5  # Update last action to reflect gripper command

                # Publish the hold position message
                self.execute_action(self.hold_position)  
                self.get_logger().info(f"Published hold position outputs: {self.hold_position}")
                
                # Log observations and actions to CSV
                timestamp = time.time()
                last_action_np = self.last_action.detach().cpu().numpy()[0] if self.last_action is not None else [None] * 8
                self.csv_writer.writerow([
                    timestamp,
                    self.joint_positions.tolist(),
                    self.joint_velocities.tolist(),
                    self.object_position.tolist(),
                    self.object_orientation.tolist(),
                    self.target_position.tolist(),
                    last_action_np,
                    self.hold_position["joint_positions"] + [self.hold_position["gripper_command"]]
                ])
                return
                
        except Exception as e:
            self.get_logger().error(f"Error running policy inference: {e}")

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
        # Add a callback to be called when the task is done
        result_future.add_done_callback(self.grasp_result_callback)

    # Callback when the grasp action completes
    def grasp_result_callback(self, future):
        """Callback triggered when the grasp action completes."""
        status = future.result().status
        self.get_logger().info(f"Grasp action completed with status: {status}")

        # Exit hold postition mode
        self.hold_position_active = False
       
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info("Grasp successful: Object is attached to the ee.")
            self.object_grasped = True
            self.gripper_goal_state = 'closed'
            self.last_action[0, -1] = 0
        else: 
            self.get_logger().warning("Grasp failed: Object is not attached to the ee.")
            self.object_grasped = False

        # Resume policy execution after
        self.start()  # Resume policy execution after grasping

    def check_state(self):
        """Regular check of the node's state"""
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
        self.get_logger().info("Policy execution paused.")

    def reset_state(self):
        """Reset all internal state variables to their initial values."""
        self.get_logger().info("Resetting internal policy runner state...")

        # Reset object state
        self.object_grasped = False
        self.object_position = np.array([0.5, 0.3, 0.055])  # Example: object back on table
        self.object_orientation = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion

        # Reset control flags
        self.hold_position_active = False
        self.running = False
        self.was_running_previously = False
        self.was_holding_previously = False

        # Reset joint state tracking
        self.joint_positions = np.array([0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741, 0.0, 0.0])
        self.joint_velocities = np.zeros(9)
        self.ee_pos = np.zeros(3)

        # Reset gripper state
        self.gripper_goal_state = 'open'

        # Reset last action
        self.last_action = torch.zeros((1, self.policy_loader.action_dim), device=self.policy_loader.device)

        # Reset hold position
        self.hold_position = None

        self.get_logger().info("Internal state successfully reset.")


    def reset_to_home(self):
        """Reset the robot to home position and open gripper."""
        self.get_logger().info("Resetting robot to home position...")
        
        # Reset internal state
        self.reset_state()

        # Send home position command to robot
        home_position = [0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741]

        processed_policy_outputs_msg = Float64MultiArray()
        processed_data = home_position + [4.0]  # Append open gripper command
        processed_policy_outputs_msg.data = processed_data
        self.policy_outputs_publisher.publish(processed_policy_outputs_msg)

        # Send gripper open command via action client
        goal_msg = Move.Goal()
        goal_msg.width = self.gripper_max_width
        goal_msg.speed = self.gripper_speed
        self.move_client.send_goal_async(goal_msg)

        self.get_logger().info("Robot reset complete. Policy execution is paused.")


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