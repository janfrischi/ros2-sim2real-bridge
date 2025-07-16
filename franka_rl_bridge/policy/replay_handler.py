#!/usr/bin/env python3
"""
Replay Handler - Manages replay data processing and comparison
"""
import json
import os
import sys
import numpy as np
from typing import Dict, List, Optional

class ReplayHandler:
    """Handles replay data loading and step-through functionality"""
    
    def __init__(self, replay_file: str):
        self.replay_file = replay_file
        self.replay_data = None
        self.replay_index = 0
        self.replay_trial = 0
        self.replay_auto = False
        self.replay_execute_actions = False
        self.replay_auto_next_trial = False
        
        self.load_replay_data()
    
    def load_replay_data(self):
        """Load observation data from JSON file"""
        try:
            if not os.path.exists(self.replay_file):
                raise FileNotFoundError(f"Replay file not found: {self.replay_file}")
            
            with open(self.replay_file, 'r') as f:
                self.replay_data = json.load(f)
            
            print(f"✅ Loaded replay data with {len(self.replay_data)} trials")
            for i, trial in enumerate(self.replay_data):
                num_obs = len(trial['observations'])
                total_steps = trial['metadata']['total_steps']
                print(f"   Trial {i}: {num_obs} observations, {total_steps} total steps")
        
        except Exception as e:
            print(f"Failed to load replay data: {e}")
            raise
    
    def parse_replay_observation(self, obs_data: Dict) -> Dict[str, np.ndarray]:
        """Convert observation data from JSON to policy input format"""
        try:
            return {
                'eef_pos': np.array(obs_data['eef_pos'], dtype=np.float32),
                'eef_quat': np.array(obs_data['eef_quat'], dtype=np.float32),
                'gripper_pos': np.array(obs_data['gripper_pos'], dtype=np.float32),
                'object': np.array(obs_data['object'], dtype=np.float32)
            }
        except Exception as e:
            print(f"Error parsing replay observation: {e}")
            return None
    
    def get_current_observation(self) -> Optional[Dict]:
        """Get current observation from replay data"""
        if not self.replay_data or self.replay_trial >= len(self.replay_data):
            return None
        
        current_trial = self.replay_data[self.replay_trial]
        observations = current_trial['observations']
        
        if self.replay_index >= len(observations):
            return None
        
        return observations[self.replay_index]
    
    def step_forward(self) -> bool:
        """Step to next observation, returns True if successful"""
        if not self.replay_data:
            return False
        
        current_trial = self.replay_data[self.replay_trial]
        observations = current_trial['observations']
        
        if self.replay_index >= len(observations):
            # End of current trial
            if self.replay_auto_next_trial and self.replay_trial + 1 < len(self.replay_data):
                print(f"🔄 Auto-switching to trial {self.replay_trial + 2}")
                self.next_trial()
                return True
            else:
                print(f"🏁 End of trial {self.replay_trial + 1} reached")
                self.replay_auto = False
                return False
        
        self.replay_index += 1
        return True
    
    def reset_trial(self):
        """Reset to beginning of current trial"""
        self.replay_index = 0
        print(f"🔄 Reset replay to beginning of trial {self.replay_trial}")
    
    def next_trial(self):
        """Switch to next trial"""
        if self.replay_trial < len(self.replay_data) - 1:
            self.replay_trial += 1
            self.reset_trial()
            print(f"➡️ Switched to trial {self.replay_trial}")
        else:
            print("Already at last trial")
    
    def prev_trial(self):
        """Switch to previous trial"""
        if self.replay_trial > 0:
            self.replay_trial -= 1
            self.reset_trial()
            print(f"⬅️ Switched to trial {self.replay_trial}")
        else:
            print("Already at first trial")
    
    def show_trial_info(self):
        """Display information about current trial"""
        if not self.replay_data:
            return
        
        current_trial = self.replay_data[self.replay_trial]
        metadata = current_trial['metadata']
        observations = current_trial['observations']
        
        print(f"\n{'='*60}")
        print(f"TRIAL {self.replay_trial} INFORMATION")
        print(f"{'='*60}")
        print(f"Task:           {metadata['task']}")
        print(f"Checkpoint:     {os.path.basename(metadata['checkpoint'])}")
        print(f"Horizon:        {metadata['horizon']}")
        print(f"Frequency:      {metadata['frequency_hz']} Hz")
        print(f"Total steps:    {metadata['total_steps']}")
        print(f"Observations:   {len(observations)}")
        print(f"Current step:   {self.replay_index}")
        print(f"Progress:       {self.replay_index}/{len(observations)-1} ({100*self.replay_index/max(1,len(observations)-1):.1f}%)")
        print(f"{'='*60}")
    
    def display_comparison(self, obs_data: Dict, policy_action: np.ndarray, recorded_action: List):
        """Display comparison between recorded and policy output"""
        step = obs_data['step']
        timestamp = obs_data['timestamp']
        
        sys.stdout.flush()
        sys.stderr.flush()
        
        print("\n" + "="*80)
        print(f"REPLAY STEP {step} (t={timestamp:.2f}s) - Trial {self.replay_trial}")
        print("="*80)
        
        # Show observation breakdown
        print("📥 INPUT OBSERVATION:")
        eef_pos = obs_data['eef_pos']
        eef_quat = obs_data['eef_quat']
        gripper_pos = obs_data['gripper_pos']
        
        print(f"   EEF Position:       [{eef_pos[0]:+7.4f}, {eef_pos[1]:+7.4f}, {eef_pos[2]:+7.4f}]")
        print(f"   EEF Quaternion:     [{eef_quat[0]:+7.4f}, {eef_quat[1]:+7.4f}, {eef_quat[2]:+7.4f}, {eef_quat[3]:+7.4f}]")
        print(f"   Gripper Pos:        [{gripper_pos[0]:+7.4f}, {gripper_pos[1]:+7.4f}]")
        
        print("\n🎬 POLICY OUTPUT:")
        print(f"   Full Action:     [{', '.join([f'{x:+7.4f}' for x in policy_action])}]")
        print(f"   EEF Pose:        [{', '.join([f'{x:+7.4f}' for x in policy_action[:7]])}]")
        print(f"   Gripper Cmd:     {policy_action[7]:+7.4f}")
        
        if recorded_action:
            print("\n📹 RECORDED ACTION:")
            print(f"   Full Action:     [{', '.join([f'{x:+7.4f}' for x in recorded_action])}]")
            print(f"   EEF Pose:        [{', '.join([f'{x:+7.4f}' for x in recorded_action[:7]])}]")
            print(f"   Gripper Cmd:     {recorded_action[7]:+7.4f}")
            
            # Calculate differences
            if len(recorded_action) == len(policy_action):
                diff = np.abs(np.array(policy_action) - np.array(recorded_action))
                print(f"\n📊 DIFFERENCES:")
                print(f"   Max Difference:  {np.max(diff):+7.4f}")
                print(f"   Mean Difference: {np.mean(diff):+7.4f}")
        
        print("="*80)
        sys.stdout.flush()
        sys.stderr.flush()