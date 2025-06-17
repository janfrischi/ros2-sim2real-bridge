#!/usr/bin/env python3
"""
Monitor observations that BC policy runner would receive
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
import numpy as np
import time
from scipy.spatial.transform import Rotation as R

class ObservationMonitor(Node):
    def __init__(self):
        super().__init__('observation_monitor')
        
        # Data storage
        self.current_joint_positions = None
        self.current_eef_pose = None
        self.current_gripper_positions = None
        self.last_update_times = {}
        
        # Object state storage - hardcoded for now (same as BC policy runner)
        self.cube_positions = {
            'cube_1': np.array([0.5, 0.2, 0.0203]),
            'cube_2': np.array([0.5, 0.4, 0.0203]),
            'cube_3': np.array([0.5, -0.2, 0.0203])
        }
        
        # Cube orientations (quaternions) - w, x, y, z format
        self.cube_quaternions = {
            'cube_1': np.array([1.0, 0.0, 0.0, 0.0]),
            'cube_2': np.array([1.0, 0.0, 0.0, 0.0]),
            'cube_3': np.array([1.0, 0.0, 0.0, 0.0])
        }

        # Environment origin (base frame reference)
        self.env_origin = np.array([0.0, 0.0, 0.0])
        
        # Setup QoS
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=1
        )
        
        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState,
            'franka/joint_states',
            self.joint_state_callback,
            qos_profile
        )
        
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
        
        # Monitor timer
        self.monitor_timer = self.create_timer(0.5, self.monitor_callback)  # Slower for readability
        
        self.get_logger().info("Observation Monitor with Object State started")
    
    def joint_state_callback(self, msg: JointState):
        # Updated joint names to match actual Franka joint names (fr3 prefix)
        franka_joint_names = ['fr3_joint1', 'fr3_joint2', 'fr3_joint3', 'fr3_joint4',
                             'fr3_joint5', 'fr3_joint6', 'fr3_joint7']
        
        joint_positions = []
        for joint_name in franka_joint_names:
            if joint_name in msg.name:
                idx = msg.name.index(joint_name)
                joint_positions.append(msg.position[idx])
        
        if len(joint_positions) == 7:
            self.current_joint_positions = np.array(joint_positions)
            self.last_update_times['joints'] = time.time()
    
    def eef_pose_callback(self, msg: PoseStamped):
        self.current_eef_pose = msg
        self.last_update_times['eef_pose'] = time.time()
    
    def gripper_state_callback(self, msg: JointState):
        # Based on gripper_action_server.cpp, gripper publishes two finger positions
        if len(msg.position) >= 2:
            self.current_gripper_positions = np.array(msg.position[:2])
            self.last_update_times['gripper'] = time.time()
    
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
        if self.current_eef_pose is None:
            return np.zeros(39)

        # Get end-effector position in the world frame
        ee_pos = np.array([
            self.current_eef_pose.pose.position.x,
            self.current_eef_pose.pose.position.y,
            self.current_eef_pose.pose.position.z
        ])

        # Get cube positions and quaternions
        cube_1_pos = self.cube_positions['cube_1'] 
        cube_1_quat = self.cube_quaternions['cube_1']

        cube_2_pos = self.cube_positions['cube_2']
        cube_2_quat = self.cube_quaternions['cube_2']

        cube_3_pos = self.cube_positions['cube_3']
        cube_3_quat = self.cube_quaternions['cube_3']

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
    
    def update_cube_position(self, cube_name: str, position: np.ndarray):
        """Update cube position (for future dynamic updates)"""
        if cube_name in self.cube_positions:
            self.cube_positions[cube_name] = position.copy()
            self.get_logger().info(f"Updated {cube_name} position: {position}")
    
    def update_cube_quaternion(self, cube_name: str, quaternion: np.ndarray):
        """Update cube quaternion (for future dynamic updates)"""
        if cube_name in self.cube_quaternions:
            self.cube_quaternions[cube_name] = quaternion.copy()
            self.get_logger().info(f"Updated {cube_name} quaternion: {quaternion}")

    def set_cube_pose_from_euler(self, cube_name: str, position: np.ndarray, 
                                 euler_angles: np.ndarray):
        """
        Set cube pose using position and Euler angles
        Args:
            cube_name: Name of the cube ('cube_1', 'cube_2', 'cube_3')
            position: 3D position [x, y, z]
            euler_angles: Euler angles [roll, pitch, yaw] in radians
        """
        if cube_name in self.cube_positions:
            self.cube_positions[cube_name] = position.copy()
            
            # Convert Euler angles to quaternion (w, x, y, z)
            r = R.from_euler('xyz', euler_angles)
            quat_scipy = r.as_quat()  # Returns [x, y, z, w]
            quat_wxyz = np.array([quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]])
            self.cube_quaternions[cube_name] = quat_wxyz
            
            self.get_logger().info(f"Set {cube_name} pose - pos: {position}, euler: {euler_angles}")
    
    def create_observation_vector(self):
        """Create the full observation vector like BC policy would (48D total)"""
        if (self.current_eef_pose is None or self.current_gripper_positions is None):
            return None
        
        # End-effector position
        eef_pos = np.array([
            self.current_eef_pose.pose.position.x,
            self.current_eef_pose.pose.position.y,
            self.current_eef_pose.pose.position.z
        ])
        
        # End-effector quaternion
        eef_quat = np.array([
            self.current_eef_pose.pose.orientation.w,
            self.current_eef_pose.pose.orientation.x,
            self.current_eef_pose.pose.orientation.y,
            self.current_eef_pose.pose.orientation.z
        ])
        
        # Gripper positions
        gripper_pos = self.current_gripper_positions
        
        # Object state (computed from cube positions and relative distances)
        object_state = self.compute_object_observations()
        
        # Full observation (3 + 4 + 2 + 39 = 48D)
        obs = np.concatenate([eef_pos, eef_quat, gripper_pos, object_state])
        return obs
    
    def print_object_state_debug(self):
        """Print detailed object state information for debugging"""
        if self.current_eef_pose is None:
            self.get_logger().warn("No EEF pose available for object state debug")
            return
            
        object_obs = self.compute_object_observations()
        
        print("\n" + "=" * 70)
        print("OBJECT STATE DEBUG")
        print("=" * 70)
        
        # Print cube positions and quaternions
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
        
        # Print gripper position
        ee_pos = np.array([
            self.current_eef_pose.pose.position.x,
            self.current_eef_pose.pose.position.y, 
            self.current_eef_pose.pose.position.z
        ])
        print(f"\nGripper Position: [{ee_pos[0]:+7.4f}, {ee_pos[1]:+7.4f}, {ee_pos[2]:+7.4f}]")
        
        # Print relative distances
        idx = 21  # Start of gripper-to-cube vectors in the 39D observation
        for i, (cube_id, cube_name) in enumerate(cube_names.items()):
            vec = object_obs[idx + i*3:idx + (i+1)*3]
            dist = np.linalg.norm(vec)
            print(f"Gripper to {cube_name}: [{vec[0]:+7.4f}, {vec[1]:+7.4f}, {vec[2]:+7.4f}] (dist: {dist:.4f}m)")
        
        print(f"\nTotal object obs shape: {object_obs.shape}")
        print("=" * 70)
    
    def monitor_callback(self):
        """Print observation status"""
        current_time = time.time()
        
        print("\n" + "="*70)
        print("BC POLICY OBSERVATION MONITOR")
        print("="*70)
        
        # Check data freshness
        for topic, last_time in self.last_update_times.items():
            age = current_time - last_time
            status = "✓" if age < 1.0 else "!" if age < 5.0 else "✗"
            print(f"{status} {topic}: {age:.2f}s ago")
        
        # Show current values
        if self.current_joint_positions is not None:
            print(f"Joint Positions: [{', '.join(f'{jp:.4f}' for jp in self.current_joint_positions)}]")
        
        if self.current_eef_pose is not None:
            pos = self.current_eef_pose.pose.position
            print(f"EEF Position: [{pos.x:.4f}, {pos.y:.4f}, {pos.z:.4f}]")
            
            quat = self.current_eef_pose.pose.orientation
            print(f"EEF Quaternion: [{quat.w:.4f}, {quat.x:.4f}, {quat.y:.4f}, {quat.z:.4f}] (w,x,y,z)")
        
        if self.current_gripper_positions is not None:
            gripper_width = sum(self.current_gripper_positions)  # Total width
            print(f"Gripper Positions: [{self.current_gripper_positions[0]:.4f}, {self.current_gripper_positions[1]:.4f}] (total width: {gripper_width:.4f}m)")
        
        # Show object state summary
        object_obs = self.compute_object_observations()
        if object_obs is not None and self.current_eef_pose is not None:
            print(f"\nObject Observations (39D): range=[{object_obs.min():.3f}, {object_obs.max():.3f}]")
            
            # Show distances to cubes
            ee_pos = np.array([
                self.current_eef_pose.pose.position.x,
                self.current_eef_pose.pose.position.y,
                self.current_eef_pose.pose.position.z
            ])
            
            cube_names = ['Blue', 'Red', 'Green']
            cube_ids = ['cube_1', 'cube_2', 'cube_3']
            
            # End-effector to cube distances
            for cube_id, cube_name in zip(cube_ids, cube_names):
                cube_pos = self.cube_positions[cube_id]
                dist = np.linalg.norm(cube_pos - ee_pos)
                print(f" End-Effector Distance to {cube_name} Cube: {dist:.4f}m")

            # Cube-to-cube distances using nested loop
            cube_pairs = [
                (0, 1, 'Blue', 'Red'),      # cube_1 to cube_2
                (1, 2, 'Red', 'Green'),     # cube_2 to cube_3  
                (0, 2, 'Blue', 'Green')     # cube_1 to cube_3
            ]
            
            for i, j, name_i, name_j in cube_pairs:
                cube_i_pos = self.cube_positions[cube_ids[i]]
                cube_j_pos = self.cube_positions[cube_ids[j]]
                dist = np.linalg.norm(cube_i_pos - cube_j_pos)
                print(f" Distance between {name_i} and {name_j} Cube: {dist:.4f}m")
        
        # Create full observation vector
        obs = self.create_observation_vector()
        if obs is not None:
            print(f"\n✓ FULL BC OBSERVATION VECTOR: shape={obs.shape} (expected: 48D)")
            print(f"  Range: [{obs.min():.4f}, {obs.max():.4f}]")
            print("  Components:")
            print("    - EEF position (3D): ✓")
            print("    - EEF quaternion (4D): ✓") 
            print("    - Gripper positions (2D): ✓")
            print("    - Object observations (39D): ✓")
        else:
            print("\n✗ Cannot create full observation - missing EEF pose or gripper data")
        
        print("="*70)
        print("Commands: 'd' for detailed object debug, 'u' for cube position updates")

def main():
    rclpy.init()
    
    monitor = ObservationMonitor()
    
    try:
        print("Starting BC Policy Observation Monitor with Object State...")
        print("This monitor shows the same 48D observation vector that BC policy would receive.")
        print("Press Ctrl+C to stop")
        rclpy.spin(monitor)
    except KeyboardInterrupt:
        print("\nStopping BC policy observation monitor")
    finally:
        rclpy.shutdown()

if __name__ == "__main__":
    main()