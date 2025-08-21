#!/usr/bin/env python3
"""
Behavior Cloning Policy Runner for Franka Robot
This node loads a trained BC policy using robomimic and runs inference on the Franka robot.
"""
import torch
import numpy as np
import argparse
import rclpy
import threading
import signal
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

# Import Functionality from utils folder
from franka_rl_bridge.imitation_learning.utils.gripper_control import GripperControlMixin
from franka_rl_bridge.imitation_learning.utils.cube_manager import CubeManagerMixin
from franka_rl_bridge.imitation_learning.utils.observations import ObservationMixin
from franka_rl_bridge.imitation_learning.utils.keyboard import KeyboardInput
from franka_rl_bridge.imitation_learning.utils.keyboard_handler import KeyboardHandlerMixin
from franka_rl_bridge.imitation_learning.utils.instructions import InstructionMixin
from franka_rl_bridge.imitation_learning.utils.policy_control import PolicyControlMixin
from franka_rl_bridge.imitation_learning.utils.robot_state import RobotStateMixin

# BCPolicy Runner Node
class BCPolicyRunner(Node, GripperControlMixin, CubeManagerMixin, ObservationMixin, InstructionMixin, KeyboardHandlerMixin, PolicyControlMixin, RobotStateMixin):
    """ROS2 Node for running Behavior Cloning policy on Franka robot"""
    def __init__(self, policy_path: str, device: str = "cpu", deterministic: bool = True, 
                 control_frequency: float = 20.0, testing_mode: bool = False):
        super().__init__('bc_policy_runner')
        
        # Store testing mode
        self.testing_mode = testing_mode
        
        # Initialize parameters
        self.device = torch.device(device)
        self.deterministic = deterministic
        self.control_frequency = control_frequency
        self.step_count = 0

        # Add trial tracking
        self.trial_id = 0
        self.current_config_name = "Default"

        # Initialize keyboard input handler
        self.keyboard = KeyboardInput()
        
        # Load policy using robomimic framework and set it to evaluation mode
        self.policy, self.ckpt_dict = self.load_policy(policy_path, device=device)

        # Initialize for new episode
        self.policy.start_episode()  
        
        # Robot state storage
        self.current_eef_pose = None
        self.current_gripper_positions = None
        self.current_jacobian = None

        # Object state storage
        self.cube_positions = {
            'cube_1': np.array([0.400, -0.200, 0.0203]),
            'cube_2': np.array([0.475, -0.046, 0.0203]),
            'cube_3': np.array([0.430, -0.279, 0.0203])
        }
        
        # Cube orientations (quaternions) - w, x, y, z format "IsaacLab expects quaternions in [w, x, y, z] format"
        self.cube_quaternions = {
            'cube_1': np.array([0.0, 0.0, 0.0, 1.0]),  # Identity quaternion
            'cube_2': np.array([0.0, 0.0, 0.0, 1.0]),  # Identity quaternion
            'cube_3': np.array([0.0, 0.0, 0.0, 1.0])   # Identity quaternion
        }

        # Dynamic object state storage
        self.cube_attached = None # None, 'cube_2', 'cube_3'
        self.last_gripper_state = 'open' # Track gripper state changes
        self.grasp_threshold = 0.055 # Width below which we consider the gripper "closed"
        self.release_threshold = 0.07 # Width above which we consider gripper "open"
        self.proximity_threshold = 0.05 # Distance threshold for grasp condition 
        self.grasp_sequence_count = 0 # 0: no grasps, 1: first grasp (cube_2) 2: second grasp (cube_3)

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

        # Camera Flags
        self.camera_cube_poses = {}
        self.camera_cube_poses_received = False
        
        # Keyboard input handler
        self.keyboard = KeyboardInput()
        
        # Callback group for allowing concurrent callbacks
        self.callback_group = ReentrantCallbackGroup()
        
        # --- Gripper Control Initialization ---
        self.gripper_goal_state = 'unknown' # 'open', 'closed', 'unknown'
        self.gripper_max_width = 0.08 # Max width for Franka Hand
        self.gripper_speed = 0.5 # Default speed (m/s)
        self.gripper_force = 40.0 # Default grasp force (N)
        self.gripper_epsilon_inner = 0.05
        self.gripper_epsilon_outer = 0.07
        self.initialize_gripper_clients()
        self.gripper_action_in_progress = False
        self.gripper_action_lock = threading.Lock()
        self.gripper_last_command_time = 0.0
        self.gripper_command_cooldown = 1.0  # 1 second between commands
        
        # Setup QoS (Quality of Service) profiles
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=1
        )

        # ------------------------------------------------------Subscribers--------------------------------------------------------
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

        # Subscribe to Jacobian Topic
        self.jacobian_sub = self.create_subscription(
            Float64MultiArray,
            '/franka/jacobian_ee',
            self.jacobian_callback,
            qos_profile,
            callback_group=self.callback_group
        )
        
        # Subscribe to cube poses from camera
        self.setup_camera_subscribers(qos_profile, self.callback_group)

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
            self.control_loop, 
            callback_group=self.callback_group,
            clock=rclpy.clock.Clock(clock_type=rclpy.clock.ClockType.STEADY_TIME)
        )
        
        # Keyboard input timer (check for keypress every 50ms)
        if self.testing_mode:
            self.keyboard_timer = self.create_timer(0.05, self.check_keyboard_input_testing)  # ADD THIS
        else:
            self.keyboard_timer = self.create_timer(0.05, self.check_keyboard_input)
        
        # Print initial instructions based on mode
        if self.testing_mode:
            self.print_instructions_testing()
        else:
            self.print_instructions()

        # Policy running state
        self.policy_running = False

    
    def control_loop(self):
        """Main control loop for normal mode"""
        if self.shutdown_requested:
            return
        if self.is_running and self.episode_active:
            try:
                self.handle_control_step()
            except Exception as e:
                self.get_logger().error(f"Control loop error: {e}")

    def handle_control_step(self):
        """Handle a single step of the control loop"""
        try:
            # STEP 1: Check for gripper state changes and handle cube attachment
            gripper_state_change = self.detect_gripper_state_change()
            # If gripper_state_change is not None, handle cube attachment logic
            if gripper_state_change:
                self.handle_cube_attachment(gripper_state_change)
        
            # STEP 2: Update cube pose if any cube is attached
            if self.cube_attached is not None and self.current_eef_pose is not None:
                self.update_attached_cube_pose()
            
            # STEP 3: Calculate manipulability index
            manipulability_index = self.calculate_manipulability_index()

            # STEP 4: Create observation for the policy (with updated cube poses)
            obs_dict = self.create_observation()

            if obs_dict is not None:

                # STEP 5: Run policy inference
                action = self.policy(obs_dict)
                # Convert action to numpy array if needed
                action_np = action if isinstance(action, np.ndarray) else action.cpu().numpy()

                # STEP 6: Save observation for analysis
                self.save_observation_to_csv(obs_dict, action_np, manipulability_index)
                
                # STEP 7: Log observation and action together
                self.log_observation_compact(obs_dict, action_np, manipulability_index)
                
                # STEP 8: Execute the action
                self.execute_action(action_np)

        except Exception as e:
            self.get_logger().error(f"Normal mode execution error: {e}")

    
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
    parser.add_argument("--testing", action="store_true",
                       help="Run in testing mode with systematic configurations")
    
    args = parser.parse_args()
    
    rclpy.init()
    
    try:
        node = BCPolicyRunner(
            policy_path=args.policy,
            device=args.device,
            deterministic=args.deterministic,
            control_frequency=args.frequency,
            testing_mode=args.testing
        )
        
        signal.signal(signal.SIGINT, lambda sig, frame: node.request_shutdown())
        rclpy.spin(node)
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        rclpy.try_shutdown()
        print("Shutdown complete.")

if __name__ == "__main__":
    main()