#!/usr/bin/env python3
"""
Core BC Policy Executor - Handles policy loading and inference
"""
import torch
import numpy as np
from typing import Dict, Optional
from robomimic.utils.file_utils import policy_from_checkpoint

class PolicyExecutor:
    """Handles BC policy loading, inference, and state management"""
    
    def __init__(self, policy_path: str, device: str = "cpu", deterministic: bool = True):
        self.device = torch.device(device)
        self.deterministic = deterministic
        self.step_count = 0
        
        # Load policy
        self.policy, self.ckpt_dict = self.load_policy(policy_path)
        self.policy.start_episode()
        
    def load_policy(self, policy_path: str):
        """Load BC policy using robomimic"""
        try:
            print(f"Loading policy from: {policy_path}")
            
            policy, ckpt_dict = policy_from_checkpoint(
                device=self.device,
                ckpt_path=policy_path,
                verbose=True
            )
            
            print("Successfully loaded policy using robomimic")
            print(f"Algorithm: {ckpt_dict.get('algo_name', 'Unknown')}")
            
            if 'shape_metadata' in ckpt_dict:
                shape_meta = ckpt_dict['shape_metadata']
                print(f"Action dimension: {shape_meta.get('ac_dim', 'Unknown')}")
                print(f"Observation keys: {list(shape_meta.get('all_shapes', {}).keys())}")
            
            return policy, ckpt_dict
            
        except Exception as e:
            print(f"Error loading policy: {e}")
            raise
    
    def predict(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Run policy inference on observation"""
        try:
            action = self.policy(obs_dict)
            action_np = action if isinstance(action, np.ndarray) else action.cpu().numpy()
            
            # Ensure 1D array
            if action_np.ndim > 1:
                action_np = action_np.squeeze()
                
            self.step_count += 1
            return action_np
            
        except Exception as e:
            print(f"Policy inference failed: {e}")
            raise
    
    def reset_episode(self):
        """Reset policy state for new episode"""
        self.policy.start_episode()
        self.step_count = 0
        print("Policy episode state reset")