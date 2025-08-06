#!/usr/bin/env python3
"""
Robot State Mixin for BC Policy Runner
Handles all robot sensor callbacks and state management.
"""
import numpy as np
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

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