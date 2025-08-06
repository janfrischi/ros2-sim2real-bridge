import numpy as np
from typing import Dict, Optional
from std_msgs.msg import Float64MultiArray

class ObservationMixin: 

    # Create observation dictionary for the policy x_t -> Input to the policy
    def create_observation(self) -> Optional[Dict[str, np.ndarray]]:
        """Create observation dictionary from current robot state for robomimic policy"""
        # Extract end-effector position
        eef_pos = np.array([
            self.current_eef_pose.pose.position.x,
            self.current_eef_pose.pose.position.y,
            self.current_eef_pose.pose.position.z
        ], dtype=np.float32)
        
        # Extract end-effector quaternion from ROS message [qx, qy, qz, qw]
        eef_quat_ros = np.array([
            self.current_eef_pose.pose.orientation.x,  # qx
            self.current_eef_pose.pose.orientation.y,  # qy  
            self.current_eef_pose.pose.orientation.z,  # qz
            self.current_eef_pose.pose.orientation.w   # qw
        ], dtype=np.float32)
        
        # TRANSFORM: Convert from ROS [qx, qy, qz, qw] to IsaacLab [qw, qx, qy, qz] for policy input
        eef_quat_sim = np.array([
            eef_quat_ros[3],  # qw
            eef_quat_ros[0],  # qx
            eef_quat_ros[1],  # qy
            eef_quat_ros[2]   # z
        ], dtype=np.float32)
        
        # Extract gripper positions 
        gripper_pos = self.current_gripper_positions.astype(np.float32)
        
        # Compute object observations
        object_state = self.compute_object_observations().astype(np.float32)
        
        # Return dictionary with proper keys for robomimic policy (numpy arrays)
        obs_dict = {
            'eef_pos': eef_pos,         # [3]
            'eef_quat': eef_quat_sim,   # [4]
            'gripper_pos': gripper_pos, # [2]
            'object': object_state      # [39]
                                        # Total: 48D observation vector
        }
        
        return obs_dict
    

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

        # Get cube positions and quaternions -> cube_2_pos, cube_3_pos are dynamic if attached
        cube_1_pos = self.cube_positions['cube_1'] 
        cube_2_pos = self.cube_positions['cube_2']
        cube_3_pos = self.cube_positions['cube_3']

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
    
    def log_observation_compact(self, obs_dict: Dict[str, np.ndarray], action_np: np.ndarray = None, manipulability_index: float = None):
        """Compact structured observation logging for full 48D vector with optional action logging and dynamic cube tracking"""
        eef_pos = obs_dict['eef_pos']           #  3D
        eef_quat = obs_dict['eef_quat']         # 4D  
        gripper_pos = obs_dict['gripper_pos']   # 2D
        object_obs = obs_dict['object']         # 39D
        
        # Calculate total observation size
        total_size = len(eef_pos) + len(eef_quat) + len(gripper_pos) + len(object_obs)
        
        print(f"\n┌{'─'*100}┐")
        print(f"│ EEF POS [0-2]   │ X:{eef_pos[0]:8.5f} │ Y:{eef_pos[1]:8.5f} │ Z:{eef_pos[2]:8.5f} │")
        
        # End-Effector Quaternion (elements 3-6)
        print(f"│ EEF QUAT [3-6]  │ W:{eef_quat[0]:8.5f} │ X:{eef_quat[1]:8.5f} │ Y:{eef_quat[2]:8.5f} │ Z:{eef_quat[3]:8.5f} │")
        
        # Gripper Position (elements 7-8)
        gripper_width = abs(gripper_pos[0]) + abs(gripper_pos[1])
        gripper_state = "OPEN" if gripper_width > 0.04 else "CLOSED"
        print(f"│ GRIPPER [7-8]   │ F1:{gripper_pos[0]:8.5f} │ F2:{gripper_pos[1]:8.5f} │ Width:{gripper_width:7.4f} │ {gripper_state:<6} │")
    
        # DYNAMIC CUBE TRACKING SECTION
        print(f"├{'─'*100}┤")
        print(f"│ DYNAMIC CUBE TRACKING STATUS │")
        print(f"├{'─'*100}┤")
        
        # Display current attachment status
        if self.cube_attached is not None:
            cube_colors = {'cube_2': '🔴 RED', 'cube_3': '🟢 GREEN'}
            attached_color = cube_colors.get(self.cube_attached, f'🟡 {self.cube_attached.upper()}')
            print(f"│ ATTACHED CUBE   │ {attached_color} CUBE ({self.cube_attached.upper()}) - Dynamic tracking ACTIVE │")
            
            # Show attachment position vs static position
            static_pos = self.cube_positions[self.cube_attached]
            print(f"│ CUBE POSITION   │ Current: [{static_pos[0]:6.3f}, {static_pos[1]:6.3f}, {static_pos[2]:6.3f}] (Dynamic) │")
            
            # Calculate how much the cube has moved from its original position
            if hasattr(self, 'cube_original_positions') and self.cube_attached in self.cube_original_positions:
                orig_pos = self.cube_original_positions[self.cube_attached]
                movement = np.linalg.norm(static_pos - orig_pos)
                print(f"│ CUBE MOVEMENT   │ Moved: {movement:.4f}m from original position │")
        else:
            print(f"│ ATTACHED CUBE   │ NONE - All cubes in static positions │")
        
        # Display grasp sequence progress
        sequence_status = {
            0: "🔄 READY - Awaiting first grasp (CUBE_2)",
            1: "🔴 PHASE 1 - CUBE_2 grasped, awaiting placement and CUBE_3 grasp",
            2: "🟢 PHASE 2 - CUBE_3 grasped, final stacking phase"
        }
        current_status = sequence_status.get(self.grasp_sequence_count, f"❓ UNKNOWN STATE ({self.grasp_sequence_count})")
        print(f"│ GRASP SEQUENCE  │ {current_status} │")
        
        # Show proximity to unattached cubes
        ee_pos_3d = np.array([eef_pos[0], eef_pos[1], eef_pos[2]])
        cube_proximities = []
        cube_names = ['cube_1', 'cube_2', 'cube_3']
        cube_colors_simple = {'cube_1': '🔵', 'cube_2': '🔴', 'cube_3': '🟢'}
        
        for cube_name in cube_names:
            if cube_name != self.cube_attached:  # Only show unattached cubes
                cube_pos = self.cube_positions[cube_name]
                distance = np.linalg.norm(ee_pos_3d - cube_pos)
                proximity_status = "NEAR" if distance < self.proximity_threshold else "FAR"
                cube_proximities.append(f"{cube_colors_simple[cube_name]}{cube_name.upper()}:{distance:.3f}m({proximity_status})")
        
        proximity_str = " │ ".join(cube_proximities)
        print(f"│ CUBE PROXIMITY  │ {proximity_str} │")
        
        print(f"├{'─'*100}┤")
        print(f"│ OBJECT STATE [9-47] - 39 ELEMENTS │")
        print(f"├{'─'*100}┤")
        
        # Parse object observations (39D breakdown)
        idx = 0
        
        # Cube 1 Position + Quaternion (elements 9-15) - with dynamic indicator
        cube1_pos = object_obs[idx:idx+3]
        cube1_quat = object_obs[idx+3:idx+7]
        dynamic_indicator1 = " (DYNAMIC)" if self.cube_attached == 'cube_1' else " (STATIC)"
        print(f"│ CUBE 1 [9-15]   │ Pos: [{cube1_pos[0]:6.3f}, {cube1_pos[1]:6.3f}, {cube1_pos[2]:6.3f}] │ Quat: [{cube1_quat[0]:5.2f}, {cube1_quat[1]:5.2f}, {cube1_quat[2]:5.2f}, {cube1_quat[3]:5.2f}]{dynamic_indicator1} │")
        idx += 7
        
        # Cube 2 Position + Quaternion (elements 16-22) - with dynamic indicator
        cube2_pos = object_obs[idx:idx+3]
        cube2_quat = object_obs[idx+3:idx+7]
        dynamic_indicator2 = " (DYNAMIC)" if self.cube_attached == 'cube_2' else " (STATIC)"
        print(f"│ CUBE 2 [16-22]  │ Pos: [{cube2_pos[0]:6.3f}, {cube2_pos[1]:6.3f}, {cube2_pos[2]:6.3f}] │ Quat: [{cube2_quat[0]:5.2f}, {cube2_quat[1]:5.2f}, {cube2_quat[2]:5.2f}, {cube2_quat[3]:5.2f}]{dynamic_indicator2} │")
        idx += 7
        
        # Cube 3 Position + Quaternion (elements 23-29) - with dynamic indicator
        cube3_pos = object_obs[idx:idx+3]
        cube3_quat = object_obs[idx+3:idx+7]
        dynamic_indicator3 = " (DYNAMIC)" if self.cube_attached == 'cube_3' else " (STATIC)"
        print(f"│ CUBE 3 [23-29]  │ Pos: [{cube3_pos[0]:6.3f}, {cube3_pos[1]:6.3f}, {cube3_pos[2]:6.3f}] │ Quat: [{cube3_quat[0]:5.2f}, {cube3_quat[1]:5.2f}, {cube3_quat[2]:5.2f}, {cube3_quat[3]:5.2f}]{dynamic_indicator3} │")
        cube3_pos = object_obs[idx:idx+3]
        cube3_quat = object_obs[idx+3:idx+7]
        dynamic_indicator3 = " (DYNAMIC)" if self.cube_attached == 'cube_3' else " (STATIC)"
        print(f"│ CUBE 3 [23-29]  │ Pos: [{cube3_pos[0]:6.3f}, {cube3_pos[1]:6.3f}, {cube3_pos[2]:6.3f}] │ Quat: [{cube3_quat[0]:5.2f}, {cube3_quat[1]:5.2f}, {cube3_quat[2]:5.2f}, {cube3_quat[3]:5.2f}]{dynamic_indicator3} │")
        idx += 7
        
        # Relative positions EEF to Cubes (elements 30-38)
        print(f"├{'─'*100}┤")
        print(f"│ EEF TO CUBE RELATIVE POSITIONS │")
        print(f"├{'─'*100}┤")
        for i in range(3):
            rel_pos = object_obs[idx:idx+3]
            distance = np.linalg.norm(rel_pos)
            element_range = f"[{30+i*3}-{32+i*3}]"
            
            # Add grasp indicators
            cube_name = f"cube_{i+1}"
            grasp_indicator = ""
            if cube_name == self.cube_attached:
                grasp_indicator = " (GRASPED)"
            elif distance < self.proximity_threshold:
                grasp_indicator = " (GRASPABLE)"
            
            print(f"│ EEF→CUBE{i+1} {element_range} │ Rel: [{rel_pos[0]:7.4f}, {rel_pos[1]:7.4f}, {rel_pos[2]:7.4f}] │ Dist: {distance:6.4f}m{grasp_indicator} │")
            idx += 3
    
        # Cube-to-Cube relative positions (elements 39-47)
        print(f"├{'─'*100}┤")
        print(f"│ CUBE TO CUBE RELATIVE POSITIONS │")
        print(f"├{'─'*100}┤")
        
        # Cube 1 to Cube 2 (elements 39-41)
        cube1_to_cube2 = object_obs[idx:idx+3]
        distance_1_2 = np.linalg.norm(cube1_to_cube2)
        stack_indicator_12 = " (STACKED)" if distance_1_2 < 0.06 else ""
        print(f"│ CUBE1→CUBE2 [39-41] │ Rel: [{cube1_to_cube2[0]:7.4f}, {cube1_to_cube2[1]:7.4f}, {cube1_to_cube2[2]:7.4f}] │ Dist: {distance_1_2:6.4f}m{stack_indicator_12} │")
        idx += 3
        
        # Cube 2 to Cube 3 (elements 42-44)
        cube2_to_cube3 = object_obs[idx:idx+3]
        distance_2_3 = np.linalg.norm(cube2_to_cube3)
        stack_indicator_23 = " (STACKED)" if distance_2_3 < 0.06 else ""
        print(f"│ CUBE2→CUBE3 [42-44] │ Rel: [{cube2_to_cube3[0]:7.4f}, {cube2_to_cube3[1]:7.4f}, {cube2_to_cube3[2]:7.4f}] │ Dist: {distance_2_3:6.4f}m{stack_indicator_23} │")
        idx += 3
        
        # Cube 1 to Cube 3 (elements 45-47)
        cube1_to_cube3 = object_obs[idx:idx+3]
        distance_1_3 = np.linalg.norm(cube1_to_cube3)
        stack_indicator_13 = " (STACKED)" if distance_1_3 < 0.06 else ""
        print(f"│ CUBE1→CUBE3 [45-47] │ Rel: [{cube1_to_cube3[0]:7.4f}, {cube1_to_cube3[1]:7.4f}, {cube1_to_cube3[2]:7.4f}] │ Dist: {distance_1_3:6.4f}m{stack_indicator_13} │")
        idx += 3
        
        # Action logging section (if action is provided)
        if action_np is not None:
            print(f"├{'─'*100}┤")
            print(f"│ POLICY ACTION OUTPUT - 8D ACTION VECTOR │")
            print(f"├{'─'*100}┤")
            
            # Ensure it's a 1D array
            if action_np.ndim > 1:
                action_np = action_np.squeeze()
            
            # Parse action components
            eef_pose_action = action_np[:7]  # [x, y, z, qw, qx, qy, qz] - IsaacLab format
            gripper_action = action_np[7]    # Gripper command
            
            # End-effector pose action
            action_pos = eef_pose_action[:3]
            action_quat = eef_pose_action[3:]  # [qw, qx, qy, qz]
            
            print(f"│ ACTION POS [0-2] │ X:{action_pos[0]:8.5f} │ Y:{action_pos[1]:8.5f} │ Z:{action_pos[2]:8.5f} │")
            print(f"│ ACTION QUAT[3-6] │ W:{action_quat[0]:8.5f} │ X:{action_quat[1]:8.5f} │ Y:{action_quat[2]:8.5f} │ Z:{action_quat[3]:8.5f} │")
            
            # Gripper action analysis with dynamic context
            gripper_cmd_clamped = np.clip(gripper_action, -1.0, 1.0)
            gripper_pos_cmd = (gripper_cmd_clamped + 1.0) * 0.04  # Maps [-1,1] to [0, 0.08]
            gripper_state_cmd = "CLOSE" if gripper_action < 0 else "OPEN"
            
            # Add context about what the gripper action might achieve
            gripper_context = ""
            if gripper_state_cmd == "CLOSE" and self.cube_attached is None:
                # Check which cube is closest
                closest_cube = None
                min_distance = float('inf')
                for cube_name in ['cube_2', 'cube_3']:  # Only check graspable cubes
                    if cube_name != self.cube_attached:
                        cube_pos = self.cube_positions[cube_name]
                        distance = np.linalg.norm(ee_pos_3d - cube_pos)
                        if distance < min_distance:
                            min_distance = distance
                            closest_cube = cube_name
            
                if closest_cube and min_distance < self.proximity_threshold:
                    gripper_context = f" (ATTEMPT GRASP {closest_cube.upper()})"
                else:
                    gripper_context = " (NO CUBE IN RANGE)"
                    
            elif gripper_state_cmd == "OPEN" and self.cube_attached is not None:
                gripper_context = f" (RELEASE {self.cube_attached.upper()})"
            
            print(f"│ ACTION GRIP [7]  │ Raw:{gripper_action:8.5f} │ Clamped:{gripper_cmd_clamped:8.5f} │ Pos:{gripper_pos_cmd:7.4f} │ Cmd:{gripper_state_cmd:<6}{gripper_context} │")
            
            # Action magnitude analysis
            pos_change_mag = np.linalg.norm(action_pos - eef_pos)
            quat_diff = np.abs(action_quat - eef_quat).sum()
            
            print(f"├{'─'*100}┤")
            print(f"│ ACTION ANALYSIS │ Pos Change: {pos_change_mag:6.4f}m │ Quat Diff: {quat_diff:6.4f} │ Gripper Δ: {gripper_action:7.4f} │")
    
        # Enhanced summary statistics with dynamic tracking info
        print(f"├{'─'*100}┤")
        eef_to_cubes_min_dist = min([np.linalg.norm(object_obs[21+i*3:24+i*3]) for i in range(3)])
        cube_to_cube_min_dist = min([distance_1_2, distance_2_3, distance_1_3])
        
        # Count how many cubes are stacked
        stacked_pairs = []
        if distance_1_2 < 0.06:
            stacked_pairs.append("1→2")
        if distance_2_3 < 0.06:
            stacked_pairs.append("2→3")
        if distance_1_3 < 0.06:
            stacked_pairs.append("1→3")
        
        stack_status = f" │ Stacked: {','.join(stacked_pairs) if stacked_pairs else 'NONE'}"
        attachment_status = f" │ Attached: {self.cube_attached.upper() if self.cube_attached else 'NONE'}"
        
        # Add manipulability logging section
        if manipulability_index is not None:
            print(f"├{'─'*100}┤")
            # Display in scientific notation with 3 significant figures
            print(f"│ ROBOT MANIPULABILITY INDEX │ Value: {manipulability_index:.3e} │")
        
        manipulability_str = f" │ Manipulability: {manipulability_index:.3e}" if manipulability_index is not None else ""
        print(f"│ SUMMARY         │ Total: {total_size} elements │ EEF height: {eef_pos[2]:6.4f}m │ Closest EEF→Cube: {eef_to_cubes_min_dist:6.4f}m │ Closest Cube→Cube: {cube_to_cube_min_dist:6.4f}m{stack_status}{attachment_status}{manipulability_str} │")
        print(f"└{'─'*100}┘")
        
        # Increment step count
        self.step_count += 1

    def save_observation_to_csv(self, obs_dict: Dict[str, np.ndarray], action_np: Optional[np.ndarray] = None, manipulability_index: float = None):
        """
        Save EEF pose observations and actions to CSV file with timestamps for dynamics analysis.
        Saves position (x,y,z), quaternion (x,y,z,w) components, and action data.
        
        Args:
            obs_dict: Observation dictionary containing eef_pos and eef_quat
            action_np: Action array [x, y, z, qw, qx, qy, qz, gripper_cmd] (optional)
        """
        try:
            # ONLY RECORD DATA WHEN POLICY IS ACTIVELY RUNNING
            if not hasattr(self, 'policy_running') or not self.policy_running:
                return  # Skip recording if policy is not running
            
            # Also skip if we're in initialization phase (step_count is very low and no action)
            if self.step_count < 2 and action_np is None:
                return  # Skip initial observation without action
            
            import csv
            import os
            from datetime import datetime
            
            # Create data directory if it doesn't exist
            data_dir = os.path.join(os.path.expanduser("~"), "bc_policy_data")
            os.makedirs(data_dir, exist_ok=True)
            
            # Initialize CSV file and header flag if not exists
            if not hasattr(self, 'csv_filename'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.csv_filename = os.path.join(data_dir, f"eef_dynamics_{timestamp}.csv")
                self.csv_file_initialized = False
            
            # Get current timestamp
            current_time = datetime.now()
            timestamp_str = current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # Include milliseconds
            
            # Extract EEF position and quaternion from observation
            eef_pos = obs_dict['eef_pos']      # [x, y, z]
            eef_quat = obs_dict['eef_quat']    # [qw, qx, qy, qz] - IsaacLab format
            
            # Convert quaternion from IsaacLab [qw, qx, qy, qz] to standard [qx, qy, qz, qw] for CSV
            eef_quat_standard = np.array([eef_quat[1], eef_quat[2], eef_quat[3], eef_quat[0]])
            
            # Prepare row data with observation
            row_data = [
                self.trial_id,           # Add trial_id as first column
                self.step_count,
                timestamp_str,
                self.current_config_name,  # Add config name
                float(eef_pos[0]),    # x
                float(eef_pos[1]),    # y  
                float(eef_pos[2]),    # z
                float(eef_quat_standard[0]),  # qx
                float(eef_quat_standard[1]),  # qy
                float(eef_quat_standard[2]),  # qz
                float(eef_quat_standard[3])   # qw
            ]
            
            # Add action data if provided
            if action_np is not None and len(action_np) >= 8:
                # Action format: [x, y, z, qw, qx, qy, qz, gripper_cmd]
                action_pos = action_np[:3]      # [x, y, z]
                action_quat_sim = action_np[3:7] # [qw, qx, qy, qz] - IsaacLab format
                action_gripper = action_np[7]   # gripper command
                
                # Convert action quaternion from IsaacLab [qw, qx, qy, qz] to standard [qx, qy, qz, qw]
                action_quat_standard = np.array([action_quat_sim[1], action_quat_sim[2], action_quat_sim[3], action_quat_sim[0]])
                
                # Add action data to row
                row_data.extend([
                    float(action_pos[0]),         # action_x
                    float(action_pos[1]),         # action_y
                    float(action_pos[2]),         # action_z
                    float(action_quat_standard[0]), # action_qx
                    float(action_quat_standard[1]), # action_qy
                    float(action_quat_standard[2]), # action_qz
                    float(action_quat_standard[3]), # action_qw
                    float(action_gripper)         # action_gripper
                ])
            else:
                # Add empty action columns if no action provided
                row_data.extend([None] * 8)  # 8 action columns
            
            # Add manipulability_index to the row data
            if manipulability_index is not None:
                row_data.append(float(manipulability_index))
            else:
                row_data.append(0.0)  # Default value if not provided
            
            # Write to CSV file
            with open(self.csv_filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header only if we haven't initialized the file yet
                if not self.csv_file_initialized:
                    header = [
                        'trial_id',          # Add trial_id as first column
                        'step_count',
                        'timestamp',
                        'config_name',       # Add config_name column
                        'eef_pos_x',
                        'eef_pos_y', 
                        'eef_pos_z',
                        'eef_quat_x',
                        'eef_quat_y',
                        'eef_quat_z',
                        'eef_quat_w',
                        'action_x',
                        'action_y',
                        'action_z',
                        'action_quat_x',
                        'action_quat_y',
                        'action_quat_z',
                        'action_quat_w',
                        'action_gripper',
                        'manipulability_index'  # New column for manipulability index
                    ]
                    writer.writerow(header)
                    self.csv_file_initialized = True
                    self.get_logger().info(f"Created EEF dynamics CSV file with action data: {self.csv_filename}")
                
                # Write data row
                writer.writerow(row_data)
        
        except Exception as e:
            self.get_logger().error(f"Error saving observation to CSV: {e}")

    def publish_observation_debug(self, obs_dict: Dict[str, np.ndarray]):
        """Publish observation for debugging purposes"""
        try:
            # Concatenate all observation components into a single array (same as policy input)
            obs_flat = np.concatenate([
                obs_dict['eef_pos'],      # [3]
                obs_dict['eef_quat'],     # [4]  
                obs_dict['gripper_pos'],  # [2]
                obs_dict['object']        # [39]
            ])  # Result: [48]
            
            # Create and publish message
            obs_msg = Float64MultiArray()
            obs_msg.data = obs_flat.tolist()
            self.observation_debug_pub.publish(obs_msg)
                
        except Exception as e:
            self.get_logger().warn(f"Error publishing observation debug: {e}")