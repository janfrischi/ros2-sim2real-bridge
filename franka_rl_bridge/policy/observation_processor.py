#!/usr/bin/env python3
"""
Observation Processor - Handles observation creation and coordinate transforms
"""
import numpy as np
from typing import Dict, Optional
from geometry_msgs.msg import PoseStamped

class ObservationProcessor:
    """Processes robot state into policy observations"""
    
    def __init__(self):
        # Environment origin (base frame reference)
        self.env_origin = np.array([0.0, 0.0, 0.0])
        
        # Object state storage
        self.cube_positions = {
            'cube_1': np.array([0.4221598207950592, -0.1940348893404007, 0.0203000009059906]),
            'cube_2': np.array([0.47585567831993103, -0.046219781041145325, 0.0203000009059906]),
            'cube_3': np.array([0.4306733310222626, -0.2792506217956543, 0.0203000009059906])
        }
        
        # Cube orientations (quaternions) - w, x, y, z format
        self.cube_quaternions = {
            'cube_1': np.array([0.0, 0.0, 0.0, 1.0]),
            'cube_2': np.array([0.0, 0.0, 0.0, 1.0]),
            'cube_3': np.array([0.0, 0.0, 0.0, 1.0])
        }
    
    def create_observation(self, eef_pose: PoseStamped, gripper_positions: np.ndarray) -> Dict[str, np.ndarray]:
        """Create observation dictionary from robot state"""
        
        # Extract end-effector position
        eef_pos = np.array([
            eef_pose.pose.position.x,
            eef_pose.pose.position.y,
            eef_pose.pose.position.z
        ], dtype=np.float32)
        
        # Extract and convert quaternion: ROS [qx,qy,qz,qw] -> IsaacLab [qw,qx,qy,qz]
        eef_quat_ros = np.array([
            eef_pose.pose.orientation.x,
            eef_pose.pose.orientation.y,
            eef_pose.pose.orientation.z,
            eef_pose.pose.orientation.w
        ], dtype=np.float32)
        
        eef_quat_sim = np.array([
            eef_quat_ros[3],  # qw
            eef_quat_ros[0],  # qx
            eef_quat_ros[1],  # qy
            eef_quat_ros[2]   # qz
        ], dtype=np.float32)
        
        # Process gripper state
        gripper_pos = gripper_positions.astype(np.float32)
        
        # Compute object observations
        object_state = self.compute_object_observations(eef_pos).astype(np.float32)
        
        return {
            'eef_pos': eef_pos,
            'eef_quat': eef_quat_sim,
            'gripper_pos': gripper_pos,
            'object': object_state
        }
    
    def compute_object_observations(self, ee_pos: np.ndarray) -> np.ndarray:
        """Compute 39D object observations matching IsaacLab structure"""
        
        # Get cube positions and quaternions
        cube_1_pos = self.cube_positions['cube_1']
        cube_2_pos = self.cube_positions['cube_2']
        cube_3_pos = self.cube_positions['cube_3']
        
        cube_1_quat = self.cube_quaternions['cube_1']
        cube_2_quat = self.cube_quaternions['cube_2']
        cube_3_quat = self.cube_quaternions['cube_3']
        
        # Compute relative positions from environment origin
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
        
        # Concatenate all observations
        object_obs = np.concatenate([
            cube_1_pos_rel,      # [3]
            cube_1_quat,         # [4]
            cube_2_pos_rel,      # [3]
            cube_2_quat,         # [4]
            cube_3_pos_rel,      # [3]
            cube_3_quat,         # [4]
            gripper_to_cube_1,   # [3]
            gripper_to_cube_2,   # [3]
            gripper_to_cube_3,   # [3]
            cube_1_to_2,         # [3]
            cube_2_to_3,         # [3]
            cube_1_to_3          # [3]
        ])
        
        return object_obs
    
    def update_cube_positions(self, positions: Dict[str, np.ndarray]):
        """Update cube positions"""
        self.cube_positions.update(positions)
    
    def update_cube_quaternions(self, quaternions: Dict[str, np.ndarray]):
        """Update cube orientations"""
        self.cube_quaternions.update(quaternions)