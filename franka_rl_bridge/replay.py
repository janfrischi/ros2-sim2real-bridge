#!/usr/bin/env python3
"""
Trajectory Replayer for ROS2
Replays actions from JSON file to cartesian position controller
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.action import ActionClient
from std_msgs.msg import Float64MultiArray
from franka_msgs.action import Homing, Move, Grasp
from action_msgs.msg import GoalStatus
import json
import sys
import time
import numpy as np
import threading
import select
import termios
import tty

class TrajectoryReplayer(Node):
    def __init__(self, json_file_path):
        super().__init__('trajectory_replayer')
        
        # Callback group for allowing concurrent callbacks
        self.callback_group = ReentrantCallbackGroup()
        
        # Gripper Control Initialization
        self.gripper_goal_state = 'unknown'  # 'open', 'closed', 'unknown'
        self.gripper_max_width = 0.08  # Max width for Franka Hand
        self.gripper_speed = 0.5  # Default speed (m/s)
        self.gripper_force = 50.0  # Default grasp force (N)
        self.gripper_epsilon_inner = 0.05  # Tolerance for successful grasp
        self.gripper_epsilon_outer = 0.07
        
        # Action clients for gripper
        self.homing_client = ActionClient(self, Homing, '/fr3_gripper/homing', callback_group=self.callback_group)
        self.move_client = ActionClient(self, Move, '/fr3_gripper/move', callback_group=self.callback_group)
        self.grasp_client = ActionClient(self, Grasp, '/fr3_gripper/grasp', callback_group=self.callback_group)
        
        # Wait for gripper action servers
        self.wait_for_action_server(self.homing_client, 'Homing')
        self.wait_for_action_server(self.move_client, 'Move')
        self.wait_for_action_server(self.grasp_client, 'Grasp')
        
        # Perform initial homing
        self.home_gripper()
        
        # Setup QoS profile
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        
        # Create publisher for cartesian position commands
        # Using Float64MultiArray to match your bc_policy_runner.py
        self.pub = self.create_publisher(
            Float64MultiArray,
            '/cartesian_position_controller/commands',
            qos_profile
        )
        
        # Load trajectory data from JSON file
        self.load_trajectory_data(json_file_path)
        
        # Wait for publisher to be ready
        time.sleep(1.0)
        
    def load_trajectory_data(self, json_file_path):
        """Load and parse the trajectory data from JSON file"""
        try:
            with open(json_file_path, 'r') as f:
                data = json.load(f)
            
            # Organize data by trials
            self.trials = []
            
            # Extract trajectory points organized by trial
            for trial_idx, trial in enumerate(data):
                if 'observations' in trial:
                    trial_points = []
                    for obs in trial['observations']:
                        if 'action' in obs:
                            action = obs['action']
                            
                            # Create trajectory point
                            point = {
                                'action': action,  # IsaacLab format [x,y,z,qw,qx,qy,qz,gripper] - will be transformed to Franka format
                                'timestamp': obs.get('timestamp', 0),
                                'step': obs.get('step', 0)
                            }
                            trial_points.append(point)
                    
                    if trial_points:  # Only add trials that have action points
                        self.trials.append({
                            'trial_index': trial_idx,
                            'points': trial_points
                        })
            
            total_points = sum(len(trial['points']) for trial in self.trials)
            self.get_logger().info(f"Loaded {len(self.trials)} trials with {total_points} total trajectory points")
            
            for i, trial in enumerate(self.trials):
                self.get_logger().info(f"Trial {i}: {len(trial['points'])} points")
            
        except Exception as e:
            self.get_logger().error(f"Error loading trajectory data: {e}")
            sys.exit(1)
    
    def create_command_message(self, action):
        """Create Float64MultiArray message from action"""
        msg = Float64MultiArray()
        
        # Transform quaternion from IsaacLab convention (qw,qx,qy,qz) to Franka convention (qx,qy,qz,qw)
        # IsaacLab: [x,y,z,qw,qx,qy,qz] -> Franka: [x,y,z,qx,qy,qz,qw]
        transformed_action = [
            action[0],  # x
            action[1],  # y  
            action[2],  # z
            action[4],  # qx (was at index 4 in IsaacLab)
            action[5],  # qy (was at index 5 in IsaacLab)
            action[6],  # qz (was at index 6 in IsaacLab)
            action[3]   # qw (was at index 3 in IsaacLab)
        ]
        
        msg.data = transformed_action
        
        return msg
    
    def wait_for_action_server(self, client, name):
        """Helper to wait for action servers"""
        self.get_logger().info(f'Waiting for {name} action server...')
        while not client.wait_for_server(timeout_sec=2.0) and rclpy.ok():
            self.get_logger().info(f'{name} action server not available, waiting again...')
        if rclpy.ok():
            self.get_logger().info(f'{name} action server found.')
        else:
            self.get_logger().error(f'ROS shutdown while waiting for {name} server.')
            raise SystemExit('ROS shutdown')

    def home_gripper(self):
        """Home the gripper"""
        goal_msg = Homing.Goal()
        self.homing_client.send_goal_async(goal_msg)
        self.gripper_goal_state = 'open'  # Assume homing opens the gripper
        self.get_logger().info("Gripper homing initiated")

    def open_gripper(self):
        """Open the gripper using the action client"""
        goal_msg = Move.Goal()
        goal_msg.width = self.gripper_max_width
        goal_msg.speed = self.gripper_speed
        self.move_client.send_goal_async(goal_msg)
        self.gripper_goal_state = 'open'
        self.get_logger().info("Gripper opening command sent")

    def close_gripper(self):
        """Close the gripper using the action client"""
        goal_msg = Grasp.Goal()
        goal_msg.width = 0.0
        goal_msg.speed = self.gripper_speed
        goal_msg.force = self.gripper_force
        goal_msg.epsilon.inner = self.gripper_epsilon_inner
        goal_msg.epsilon.outer = self.gripper_epsilon_outer
        self.grasp_client.send_goal_async(goal_msg)
        self.gripper_goal_state = 'closed'
        self.get_logger().info("Gripper closing command sent")

    def control_gripper_from_action(self, gripper_command):
        """Control gripper based on action command (adapted from bc_policy_runner.py)"""
        try:
            # Determine desired gripper state based on command
            desired_gripper_state = 'closed' if gripper_command < 0 else 'open'
            
            # Execute gripper action if state has changed
            if desired_gripper_state != self.gripper_goal_state:
                if desired_gripper_state == 'open':
                    self.open_gripper()
                elif desired_gripper_state == 'closed':
                    self.close_gripper()
                    
        except Exception as e:
            self.get_logger().error(f"Error in gripper control: {e}")
    
    def wait_for_space_key(self):
        """Wait for space key press to continue"""
        print("\nPress SPACE to start the next trial (or 'q' to quit)...")
        
        try:
            # Save terminal settings
            old_settings = termios.tcgetattr(sys.stdin)
            
            try:
                # Set terminal to raw mode
                tty.setraw(sys.stdin.fileno())
                
                while True:
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        key = sys.stdin.read(1)
                        if key == ' ':  # Space key
                            print("Starting trial...")
                            return True
                        elif key.lower() == 'q':  # Quit
                            print("Quitting...")
                            return False
                        elif key == '\x03':  # Ctrl+C
                            raise KeyboardInterrupt
                    
                    # Check if ROS is still ok
                    if not rclpy.ok():
                        return False
                        
            finally:
                # Restore terminal settings
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                
        except (termios.error, OSError):
            # Fallback for environments where terminal control isn't available
            print("Terminal control not available. Press Enter to continue...")
            try:
                input()
                return True
            except (EOFError, KeyboardInterrupt):
                return False
    
    def replay_single_trial(self, trial, use_original_timing=True, speed_factor=1.0):
        """Replay a single trial"""
        points_to_replay = trial['points']
        trial_idx = trial['trial_index']
        
        self.get_logger().info(f"Starting trial {trial_idx} with {len(points_to_replay)} points...")
        
        for i, point in enumerate(points_to_replay):
            if not rclpy.ok():
                break
            
            # Get original action (8D: [x,y,z,qw,qx,qy,qz,gripper])
            original_action = point['action']
            
            # Create and publish 7D command message (pose only)
            cmd_msg = self.create_command_message(original_action)
            self.pub.publish(cmd_msg)
            
            # Control gripper based on the 8th element (gripper command)
            if len(original_action) >= 8:
                gripper_command = original_action[7]
                self.control_gripper_from_action(gripper_command)
            
            # Log progress (showing original IsaacLab format for reference)
            action = point['action']
            gripper_cmd = action[7] if len(action) >= 8 else 0.0
            gripper_state = "CLOSE" if gripper_cmd < 0 else "OPEN"
            
            self.get_logger().info(
                f"Trial {trial_idx}, Step {i+1}/{len(points_to_replay)}: "
                f"Pos=[{action[0]:.3f}, {action[1]:.3f}, {action[2]:.3f}] "
                f"Quat_Isaac=[qw:{action[3]:.3f}, qx:{action[4]:.3f}, qy:{action[5]:.3f}, qz:{action[6]:.3f}] "
                f"-> Quat_Franka=[qx:{action[4]:.3f}, qy:{action[5]:.3f}, qz:{action[6]:.3f}, qw:{action[3]:.3f}] "
                f"Gripper={gripper_cmd:.3f} ({gripper_state})"
            )
            
            # Handle timing between waypoints
            if use_original_timing and i < len(points_to_replay) - 1:
                # Use original timing from the data
                current_timestamp = point['timestamp']
                next_timestamp = points_to_replay[i + 1]['timestamp']
                sleep_duration = (next_timestamp - current_timestamp) / speed_factor
                
                if sleep_duration > 0:
                    time.sleep(sleep_duration)
                else:
                    time.sleep(0.05)  # Minimum 20Hz
            else:
                # Use fixed rate if no timing information
                time.sleep(0.05)  # 20 Hz to match your control frequency
        
        self.get_logger().info(f"Trial {trial_idx} completed!")
    
    def replay_trajectory(self, use_original_timing=True, speed_factor=1.0):
        """Replay the trajectory by publishing commands, waiting for space key between trials"""
        if not self.trials:
            self.get_logger().error("No trials to replay")
            return
        
        self.get_logger().info(f"Ready to replay {len(self.trials)} trials...")
        self.get_logger().info("Control instructions:")
        self.get_logger().info("  - Press SPACE to start each trial")
        self.get_logger().info("  - Press 'q' to quit")
        self.get_logger().info("  - Press Ctrl+C to interrupt")
        
        try:
            for trial in self.trials:
                # Wait for user input before starting each trial
                if not self.wait_for_space_key():
                    break
                
                # Replay the trial
                self.replay_single_trial(trial, use_original_timing, speed_factor)
                
                self.get_logger().info(f"Trial {trial['trial_index']} finished.")
        
        except KeyboardInterrupt:
            self.get_logger().info("Trajectory replay interrupted by user")
        
        self.get_logger().info("All trials completed!")

def main():
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python3 replay_trajectory.py <path_to_json_file> [speed_factor] [use_timing]")
        print("  speed_factor: 1.0 for original speed, 0.5 for half speed, 2.0 for double speed")
        print("  use_timing: true/false for using original timestamps")
        sys.exit(1)
    
    json_file_path = sys.argv[1]
    speed_factor = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    use_timing = sys.argv[3].lower() == 'true' if len(sys.argv) > 3 else True
    
    # Initialize ROS2
    rclpy.init()
    
    try:
        # Create replayer node
        replayer = TrajectoryReplayer(json_file_path)
        
        # Create a simple executor for this single node
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(replayer)
        
        # Start replay in a separate thread or just run it directly
        try:
            replayer.replay_trajectory(
                use_original_timing=use_timing, 
                speed_factor=speed_factor
            )
        except KeyboardInterrupt:
            replayer.get_logger().info("Trajectory replay interrupted by user")
        
    except Exception as e:
        print(f"Error during trajectory replay: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if 'replayer' in locals():
            try:
                # Destroy action clients
                replayer.homing_client.destroy()
                replayer.move_client.destroy()
                replayer.grasp_client.destroy()
                replayer.get_logger().info("Gripper action clients destroyed.")
            except:
                pass
            replayer.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()