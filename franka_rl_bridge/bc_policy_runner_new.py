#!/usr/bin/env python3
"""
BC Policy Runner - Simplified main class using modular components
"""
import torch
import numpy as np
import argparse
import sys
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from franka_msgs.action import Homing, Move, Grasp

# Import modular components
from policy.policy_executor import PolicyExecutor
from policy.observation_processor import ObservationProcessor
from policy.replay_handler import ReplayHandler
from control.environment_manager import EnvironmentManager
from control.interactive_controller import InteractiveController
from utils.data_logger import DataLogger

class BCPolicyRunner(Node):
    """Simplified BC Policy Runner using composition"""
    
    def __init__(self, policy_path: str, device: str = "cpu", deterministic: bool = True,
                 control_frequency: float = 20.0, replay_file: str = None):
        super().__init__('bc_policy_runner')
        
        # Initialize modular components
        self.policy_executor = PolicyExecutor(policy_path, device, deterministic)
        self.obs_processor = ObservationProcessor()
        self.env_manager = EnvironmentManager()
        self.data_logger = DataLogger()
        self.controller = InteractiveController()
        
        # Replay mode setup
        self.replay_mode = replay_file is not None
        if self.replay_mode:
            self.replay_handler = ReplayHandler(replay_file)
        
        # Robot state
        self.current_eef_pose = None
        self.current_gripper_positions = None
        
        # Control flags
        self.is_running = False
        self.shutdown_requested = False
        self.control_frequency = control_frequency
        
        # Setup ROS2 components
        self.setup_ros2_components()
        
        # Print instructions
        self.controller.print_instructions(
            self.replay_mode, 
            self.replay_handler.replay_data if self.replay_mode else None
        )
    
    def setup_ros2_components(self):
        """Setup ROS2 publishers, subscribers, and action clients"""
        # QoS profile
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=1
        )
        
        self.callback_group = ReentrantCallbackGroup()
        
        # Subscribers
        self.eef_pose_sub = self.create_subscription(
            PoseStamped, '/franka_robot_state_broadcaster/current_pose',
            self.eef_pose_callback, qos_profile, callback_group=self.callback_group
        )
        
        self.gripper_state_sub = self.create_subscription(
            JointState, '/fr3_gripper/joint_states',
            self.gripper_state_callback, qos_profile, callback_group=self.callback_group
        )
        
        # Publishers
        self.pose_command_pub = self.create_publisher(
            Float64MultiArray, '/cartesian_position_controller/commands', qos_profile
        )
        
        # Action clients for gripper control
        self.setup_gripper_clients()
        
        # Control timer
        self.control_timer = self.create_timer(
            1.0 / self.control_frequency, self.control_loop,
            callback_group=self.callback_group
        )
        
        # Keyboard input timer
        self.keyboard_timer = self.create_timer(0.05, self.check_keyboard_input)
    
    def setup_gripper_clients(self):
        """Setup gripper action clients"""
        self.gripper_goal_state = 'open'
        self.homing_client = ActionClient(self, Homing, '/fr3_gripper/homing', callback_group=self.callback_group)
        self.move_client = ActionClient(self, Move, '/fr3_gripper/move', callback_group=self.callback_group)
        self.grasp_client = ActionClient(self, Grasp, '/fr3_gripper/grasp', callback_group=self.callback_group)
        
        # Wait for servers and home gripper
        self.wait_for_gripper_servers()
        self.home_gripper()
    
    def wait_for_gripper_servers(self):
        """Wait for gripper action servers"""
        clients = [
            (self.homing_client, 'Homing'),
            (self.move_client, 'Move'),
            (self.grasp_client, 'Grasp')
        ]
        
        for client, name in clients:
            self.get_logger().info(f'Waiting for {name} action server...')
            while not client.wait_for_server(timeout_sec=2.0) and rclpy.ok():
                self.get_logger().info(f'{name} action server not available, waiting...')
    
    def control_loop(self):
        """Main control loop"""
        if self.shutdown_requested:
            return
        
        try:
            if self.replay_mode and self.replay_handler.replay_auto:
                self.handle_replay_execution()
            elif not self.replay_mode and self.is_running:
                self.handle_normal_execution()
        except Exception as e:
            self.get_logger().error(f"Control loop error: {e}")
    
    def handle_normal_execution(self):
        """Handle normal policy execution"""
        if self.current_eef_pose is None or self.current_gripper_positions is None:
            return
        
        # Create observation
        obs_dict = self.obs_processor.create_observation(
            self.current_eef_pose, self.current_gripper_positions
        )
        
        # Run policy inference
        action_np = self.policy_executor.predict(obs_dict)
        
        # Log data
        self.data_logger.save_observation_to_csv(obs_dict, action_np)
        self.data_logger.log_observation_compact(obs_dict, action_np)
        
        # Execute action
        self.execute_action(action_np)
    
    def handle_replay_execution(self):
        """Handle replay mode execution"""
        obs_data = self.replay_handler.get_current_observation()
        if obs_data is None:
            return
        
        # Parse observation
        obs_dict = self.replay_handler.parse_replay_observation(obs_data)
        if obs_dict is None:
            return
        
        # Run policy inference
        action_np = self.policy_executor.predict(obs_dict)
        
        # Execute action if enabled
        if self.replay_handler.replay_execute_actions:
            self.execute_action(action_np)
        
        # Display comparison
        recorded_action = obs_data.get('action', [])
        self.replay_handler.display_comparison(obs_data, action_np, recorded_action)
        
        # Step forward
        self.replay_handler.step_forward()
    
    def execute_action(self, action_np: np.ndarray):
        """Execute robot action"""
        try:
            if action_np.ndim > 1:
                action_np = action_np.squeeze()
            
            # Parse action
            eef_pose = action_np[:7]  # [x, y, z, qw, qx, qy, qz]
            gripper_command = action_np[7]
            
            # Convert quaternion: IsaacLab [qw,qx,qy,qz] -> ROS [qx,qy,qz,qw]
            position = eef_pose[:3]
            quaternion_sim = eef_pose[3:]
            quaternion_ros = np.array([
                quaternion_sim[1], quaternion_sim[2], quaternion_sim[3], quaternion_sim[0]
            ])
            
            # Normalize quaternion
            quat_norm = np.linalg.norm(quaternion_ros)
            if quat_norm > 0:
                quaternion_ros = quaternion_ros / quat_norm
            
            # Handle gripper
            desired_gripper_state = 'closed' if gripper_command < 0 else 'open'
            if desired_gripper_state != self.gripper_goal_state:
                if desired_gripper_state == 'open':
                    self.open_gripper()
                elif desired_gripper_state == 'closed':
                    self.close_gripper()
            
            # Publish pose command
            cartesian_pose = np.concatenate([position, quaternion_ros])
            pose_msg = Float64MultiArray()
            pose_msg.data = cartesian_pose.tolist()
            self.pose_command_pub.publish(pose_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error executing action: {e}")
    
    def check_keyboard_input(self):
        """Handle keyboard input"""
        try:
            command = self.controller.check_input(self.replay_mode)
            if command:
                self.handle_command(command)
        except KeyboardInterrupt:
            self.shutdown_requested = True
            raise
    
    def handle_command(self, command: str):
        """Handle user commands"""
        if command == 'quit':
            self.shutdown_requested = True
            raise KeyboardInterrupt("User requested shutdown")
        
        elif command == 'start_stop_policy':
            if not self.is_running:
                self.start_policy()
            
        elif command == 'stop_policy':
            self.stop_policy()
            
        elif command == 'reset_to_home':
            self.reset_to_home()
            
        elif command == 'toggle_gripper':
            self.toggle_gripper_manual()
            
        elif command == 'random_spawn':
            new_positions = self.env_manager.randomly_spawn_cubes()
            self.obs_processor.update_cube_positions(new_positions)
            self.policy_executor.reset_episode()
            
        elif command.startswith('spawn_'):
            pattern = command.replace('spawn_', '')
            new_positions = self.env_manager.spawn_cubes_in_pattern(pattern)
            if new_positions:
                self.obs_processor.update_cube_positions(new_positions)
                self.policy_executor.reset_episode()
        
        # Replay-specific commands
        elif self.replay_mode:
            self.handle_replay_command(command)
    
    def handle_replay_command(self, command: str):
        """Handle replay-specific commands"""
        if command == 'replay_step':
            self.replay_handler.step_forward()
        elif command == 'toggle_auto_step':
            self.replay_handler.replay_auto = not self.replay_handler.replay_auto
            status = "ENABLED" if self.replay_handler.replay_auto else "DISABLED"
            print(f"\n🔄 Auto-step mode: {status}")
        elif command == 'toggle_action_execution':
            self.replay_handler.replay_execute_actions = not self.replay_handler.replay_execute_actions
            status = "ENABLED" if self.replay_handler.replay_execute_actions else "DISABLED"
            print(f"\n🤖 Action execution: {status}")
        elif command == 'replay_reset':
            self.replay_handler.reset_trial()
            self.policy_executor.reset_episode()
        elif command == 'next_trial':
            self.replay_handler.next_trial()
            self.policy_executor.reset_episode()
        elif command == 'prev_trial':
            self.replay_handler.prev_trial()
            self.policy_executor.reset_episode()
        elif command == 'show_trial_info':
            self.replay_handler.show_trial_info()
    
    # Gripper control methods
    def home_gripper(self):
        """Home the gripper"""
        goal_msg = Homing.Goal()
        self.homing_client.send_goal_async(goal_msg)
        self.gripper_goal_state = 'open'
    
    def open_gripper(self):
        """Open the gripper"""
        goal_msg = Move.Goal()
        goal_msg.width = 0.08
        goal_msg.speed = 0.5
        self.move_client.send_goal_async(goal_msg)
        self.gripper_goal_state = 'open'
    
    def close_gripper(self):
        """Close the gripper"""
        goal_msg = Grasp.Goal()
        goal_msg.width = 0.0
        goal_msg.speed = 0.5
        goal_msg.force = 50.0
        goal_msg.epsilon.inner = 0.05
        goal_msg.epsilon.outer = 0.07
        self.grasp_client.send_goal_async(goal_msg)
        self.gripper_goal_state = 'closed'
    
    def toggle_gripper_manual(self):
        """Toggle gripper state manually"""
        if self.gripper_goal_state == 'open':
            self.close_gripper()
            print("Manual gripper: CLOSING")
        else:
            self.open_gripper()
            print("Manual gripper: OPENING")
    
    # Robot control methods
    def start_policy(self):
        """Start policy execution"""
        self.is_running = True
        self.policy_executor.reset_episode()
        print("Policy started")
    
    def stop_policy(self):
        """Stop policy execution"""
        self.is_running = False
        self.policy_executor.reset_episode()
        print("Policy stopped")
    
    def reset_to_home(self):
        """Reset robot to home position"""
        was_running = self.is_running
        if self.is_running:
            self.stop_policy()
        
        print("Resetting to home position...")
        
        # Safe home position
        home_position = np.array([0.46, 0.0, 0.266])
        home_quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # [qx, qy, qz, qw]
        
        home_pose = np.concatenate([home_position, home_quaternion])
        pose_msg = Float64MultiArray()
        pose_msg.data = home_pose.tolist()
        self.pose_command_pub.publish(pose_msg)
        
        self.open_gripper()
        self.policy_executor.reset_episode()
        
        print("Robot homed")
    
    # ROS2 callbacks
    def eef_pose_callback(self, msg: PoseStamped):
        """End-effector pose callback"""
        self.current_eef_pose = msg
    
    def gripper_state_callback(self, msg: JointState):
        """Gripper state callback"""
        finger_1_pos = msg.position[0]
        finger_2_pos = -msg.position[1]
        self.current_gripper_positions = np.array([finger_1_pos, finger_2_pos])
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            self.controller.restore_terminal()
            self.homing_client.destroy()
            self.move_client.destroy()
            self.grasp_client.destroy()
        except:
            pass

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="BC Policy Runner for Franka Robot")
    parser.add_argument("--policy", type=str, required=True, help="Path to BC policy (.pth)")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--deterministic", action="store_true", help="Deterministic inference")
    parser.add_argument("--frequency", type=float, default=20.0, help="Control frequency (Hz)")
    parser.add_argument("--replay", type=str, default=None, help="Replay JSON file")
    
    args = parser.parse_args()
    
    rclpy.init()
    
    node = None
    try:
        node = BCPolicyRunner(
            policy_path=args.policy,
            device=args.device,
            deterministic=args.deterministic,
            control_frequency=args.frequency,
            replay_file=args.replay
        )
        
        executor = rclpy.executors.MultiThreadedExecutor()
        executor.add_node(node)
        
        try:
            executor.spin()
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            executor.shutdown()
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if node:
            try:
                node.cleanup()
                node.destroy_node()
            except:
                pass
        
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except:
            pass
        
        print("Shutdown complete.")

if __name__ == "__main__":
    main()