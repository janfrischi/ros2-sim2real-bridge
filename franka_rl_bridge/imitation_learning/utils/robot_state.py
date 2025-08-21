#!/usr/bin/env python3
"""
Robot State Mixin for BC Policy Runner
Handles all robot sensor callbacks and state management.
"""
import numpy as np
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

class RobotStateMixin:
    """Mixin class for handling robot sensor callbacks and state management"""
    
    def eef_pose_callback(self, msg: PoseStamped):
        """Callback for end-effector pose updates"""
        self.current_eef_pose = msg  # Quaternion is in x, y, z, w format

    def jacobian_callback(self, msg: Float64MultiArray):
        """Callback for jacobian updates with debugging"""
        if len(msg.data) != 42:  # 6x7 = 42 elements
            self.get_logger().warn(f"Expected 42 Jacobian elements, got {len(msg.data)}")
            return
        
        # Reshape Jacobian (column-major order as published by the controller)
        self.current_jacobian = np.array(msg.data).reshape((6, 7), order='F')

    def gripper_state_callback(self, msg: JointState):
        """Callback for gripper state updates"""
        # Gripper should have symmetric but opposite values: [+value, -value] "IsaacLab Convention"
        finger_1_pos = msg.position[0]  # First finger (positive)
        finger_2_pos = -msg.position[1] # Second finger (negative)
        self.current_gripper_positions = np.array([finger_1_pos, finger_2_pos])

    def calculate_manipulability_index(self):
        """Calculate Yoshikawa's manipulability measure: sqrt(det(J * J^T))"""
        if self.current_jacobian is None:
            return 0.0
        
        try:
            # Verify Jacobian data
            if not np.isfinite(self.current_jacobian).all():
                self.get_logger().warn("Jacobian contains NaN or infinite values")
                return 0.0
            
            # Yoshikawa's manipulability: sqrt(det(J * J^T))
            # This gives the volume of the manipulability ellipsoid
            JJT = self.current_jacobian @ self.current_jacobian.T  # 6x6 matrix
            det_JJT = np.linalg.det(JJT)
            
            # Take square root and ensure non-negative
            manipulability_index = np.sqrt(max(0.0, det_JJT))
            
            return float(manipulability_index)
            
        except Exception as e:
            self.get_logger().error(f"Error calculating manipulability index: {e}")
            return 0.0
        
        
    def cube_poses_callback(self, msg):
        """Callback for camera-detected cube poses"""
        try:
            # Expect 9 elements: [cube_1_x, cube_1_y, cube_1_z, cube_2_x, cube_2_y, cube_2_z, cube_3_x, cube_3_y, cube_3_z]
            if len(msg.data) != 9:
                self.get_logger().warn(f"Expected 9 elements for 3 cube positions, got {len(msg.data)}")
                return
            
            # Extract positions for each cube
            cube_1_position = np.array([msg.data[0], msg.data[1], msg.data[2]])
            cube_2_position = np.array([msg.data[3], msg.data[4], msg.data[5]])
            cube_3_position = np.array([msg.data[6], msg.data[7], msg.data[8]])
            
            # Identity quaternion for all cubes [w, x, y, z] = [1, 0, 0, 0] in IsaacLab format
            identity_quaternion = np.array([0.0, 0.0, 0.0, 1.0])
            
            # Store camera-detected poses for all cubes
            self.camera_cube_poses = {
                'cube_1': {
                    'position': cube_1_position,
                    'quaternion': identity_quaternion
                },
                'cube_2': {
                    'position': cube_2_position, 
                    'quaternion': identity_quaternion
                },
                'cube_3': {
                    'position': cube_3_position,
                    'quaternion': identity_quaternion
                }
            }
            
            self.camera_poses_received = True
            
            self.get_logger().debug(f"Received cube poses from camera:")
            self.get_logger().debug(f"  Cube 1: {cube_1_position}")
            self.get_logger().debug(f"  Cube 2: {cube_2_position}")
            self.get_logger().debug(f"  Cube 3: {cube_3_position}")
            
        except Exception as e:
            self.get_logger().error(f"Error processing cube poses from camera: {e}")
    
    def setup_camera_subscribers(self, qos_profile, callback_group):
        """Setup subscribers for camera-detected cube poses"""
        # Initialize camera data storage
        self.camera_cube_poses = {}
        self.camera_poses_received = False
        self.expected_cubes = ['cube_1', 'cube_2', 'cube_3']
        self.received_cubes = set()
        
        # Create a single subscriber to the unified topic
        self.cube_subscriber = self.create_subscription(
            PoseStamped,
            '/perception/object_pose',
            self.unified_cube_pose_callback,
            qos_profile,
            callback_group=callback_group
        )
        
        self.get_logger().info("Camera cube pose subscriber initialized for /perception/object_pose")

    def unified_cube_pose_callback(self, msg):
        """Callback for unified cube pose messages from /perception/object_pose"""
        try:
            # Extract cube identity from frame_id
            # Expected format: "panda_link0_cube_X_idY" where X is cube number (1,2,3)
            frame_id = msg.header.frame_id
            
            # Parse frame_id to extract cube number
            cube_name = None
            if "cube_1" in frame_id:
                cube_name = "cube_1"
            elif "cube_2" in frame_id:
                cube_name = "cube_2"
            elif "cube_3" in frame_id:
                cube_name = "cube_3"
            else:
                self.get_logger().warn(f"Unknown frame_id format: {frame_id}")
                return
            
            # Extract position from PoseStamped message
            position = np.array([
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z
            ])
            
            # Use identity quaternion [w, x, y, z] = [1, 0, 0, 0] in IsaacLab format
            quaternion = np.array([0.0, 0.0, 0.0, 1.0])
            
            # Store camera-detected pose for this cube
            self.camera_cube_poses[cube_name] = {
                'position': position,
                'quaternion': quaternion
            }
            
            # Track which cubes we've received
            self.received_cubes.add(cube_name)
            
            # Mark as received if we have all cubes
            if len(self.received_cubes) >= len(self.expected_cubes):
                self.camera_poses_received = True
            
            self.get_logger().debug(f"Received pose for {cube_name} from frame_id {frame_id}: pos={position}, quat={quaternion}")
            
        except Exception as e:
            self.get_logger().error(f"Error processing unified cube pose: {e}")