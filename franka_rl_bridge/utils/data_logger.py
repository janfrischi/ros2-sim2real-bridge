#!/usr/bin/env python3
"""
Data Logger - Handles CSV logging and data persistence
"""
import csv
import os
import numpy as np
from datetime import datetime
from typing import Dict, Optional

class DataLogger:
    """Handles CSV data logging for policy execution"""
    
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            self.data_dir = os.path.join(os.path.expanduser("~"), "bc_policy_data")
        else:
            self.data_dir = data_dir
            
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.csv_filename = None
        self.csv_file_initialized = False
        self.step_count = 0
    
    def initialize_csv_file(self):
        """Initialize CSV file with timestamp"""
        if self.csv_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.csv_filename = os.path.join(self.data_dir, f"eef_dynamics_{timestamp}.csv")
            self.csv_file_initialized = False
            print(f"Initialized CSV logging: {self.csv_filename}")
    
    def save_observation_to_csv(self, obs_dict: Dict[str, np.ndarray], action_np: Optional[np.ndarray] = None):
        """Save EEF pose observations and actions to CSV"""
        try:
            if self.csv_filename is None:
                self.initialize_csv_file()
            
            # Get current timestamp
            current_time = datetime.now()
            timestamp_str = current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            
            # Extract EEF data
            eef_pos = obs_dict['eef_pos']
            eef_quat = obs_dict['eef_quat']
            
            # Convert quaternion from IsaacLab [qw,qx,qy,qz] to standard [qx,qy,qz,qw]
            eef_quat_standard = np.array([eef_quat[1], eef_quat[2], eef_quat[3], eef_quat[0]])
            
            # Prepare row data
            row_data = [
                timestamp_str,
                self.step_count,
                float(eef_pos[0]), float(eef_pos[1]), float(eef_pos[2]),
                float(eef_quat_standard[0]), float(eef_quat_standard[1]),
                float(eef_quat_standard[2]), float(eef_quat_standard[3])
            ]
            
            # Add action data if provided
            if action_np is not None and len(action_np) >= 8:
                action_pos = action_np[:3]
                action_quat_sim = action_np[3:7]  # [qw,qx,qy,qz]
                action_gripper = action_np[7]
                
                # Convert action quaternion
                action_quat_standard = np.array([
                    action_quat_sim[1], action_quat_sim[2], 
                    action_quat_sim[3], action_quat_sim[0]
                ])
                
                row_data.extend([
                    float(action_pos[0]), float(action_pos[1]), float(action_pos[2]),
                    float(action_quat_standard[0]), float(action_quat_standard[1]),
                    float(action_quat_standard[2]), float(action_quat_standard[3]),
                    float(action_gripper)
                ])
            else:
                row_data.extend([None] * 8)
            
            # Write to CSV
            with open(self.csv_filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header if first time
                if not self.csv_file_initialized:
                    header = [
                        'timestamp', 'step_count',
                        'eef_pos_x', 'eef_pos_y', 'eef_pos_z',
                        'eef_quat_x', 'eef_quat_y', 'eef_quat_z', 'eef_quat_w',
                        'action_x', 'action_y', 'action_z',
                        'action_quat_x', 'action_quat_y', 'action_quat_z', 'action_quat_w',
                        'action_gripper'
                    ]
                    writer.writerow(header)
                    self.csv_file_initialized = True
                
                writer.writerow(row_data)
                
            self.step_count += 1
            
        except Exception as e:
            print(f"Error saving observation to CSV: {e}")
    
    def log_observation_compact(self, obs_dict: Dict[str, np.ndarray], action_np: np.ndarray = None):
        """Compact observation logging"""
        eef_pos = obs_dict['eef_pos']
        eef_quat = obs_dict['eef_quat']
        gripper_pos = obs_dict['gripper_pos']
        object_obs = obs_dict['object']
        
        print(f"\n┌{'─'*100}┐")
        print(f"│ STEP {self.step_count:<6} │ FULL 48D OBSERVATION VECTOR │")
        print(f"├{'─'*100}┤")
        
        # EEF Position
        print(f"│ EEF POS [0-2]   │ X:{eef_pos[0]:8.5f} │ Y:{eef_pos[1]:8.5f} │ Z:{eef_pos[2]:8.5f} │")
        
        # EEF Quaternion
        print(f"│ EEF QUAT [3-6]  │ W:{eef_quat[0]:8.5f} │ X:{eef_quat[1]:8.5f} │ Y:{eef_quat[2]:8.5f} │ Z:{eef_quat[3]:8.5f} │")
        
        # Gripper
        gripper_width = abs(gripper_pos[0]) + abs(gripper_pos[1])
        gripper_state = "OPEN" if gripper_width > 0.04 else "CLOSED"
        print(f"│ GRIPPER [7-8]   │ F1:{gripper_pos[0]:8.5f} │ F2:{gripper_pos[1]:8.5f} │ Width:{gripper_width:7.4f} │ {gripper_state:<6} │")
        
        # Object state summary
        print(f"├{'─'*100}┤")
        print(f"│ OBJECT STATE [9-47] - 39 ELEMENTS │")
        print(f"├{'─'*100}┤")
        
        # Cube positions
        for i in range(3):
            cube_pos = object_obs[i*7:i*7+3]
            cube_quat = object_obs[i*7+3:(i+1)*7]
            print(f"│ CUBE {i+1} [{9+i*7}-{15+i*7}]   │ Pos: [{cube_pos[0]:6.3f}, {cube_pos[1]:6.3f}, {cube_pos[2]:6.3f}] │ Quat: [{cube_quat[0]:5.2f}, {cube_quat[1]:5.2f}, {cube_quat[2]:5.2f}, {cube_quat[3]:5.2f}] │")
        
        # Action output if provided
        if action_np is not None:
            print(f"├{'─'*100}┤")
            print(f"│ POLICY ACTION OUTPUT - 8D ACTION VECTOR │")
            print(f"├{'─'*100}┤")
            
            if action_np.ndim > 1:
                action_np = action_np.squeeze()
            
            action_pos = action_np[:3]
            action_quat = action_np[3:7]
            gripper_action = action_np[7]
            
            print(f"│ ACTION POS [0-2] │ X:{action_pos[0]:8.5f} │ Y:{action_pos[1]:8.5f} │ Z:{action_pos[2]:8.5f} │")
            print(f"│ ACTION QUAT[3-6] │ W:{action_quat[0]:8.5f} │ X:{action_quat[1]:8.5f} │ Y:{action_quat[2]:8.5f} │ Z:{action_quat[3]:8.5f} │")
            
            gripper_state_cmd = "CLOSE" if gripper_action < 0 else "OPEN"
            print(f"│ ACTION GRIP [7]  │ Raw:{gripper_action:8.5f} │ Cmd:{gripper_state_cmd:<6} │")
        
        print(f"└{'─'*100}┘")