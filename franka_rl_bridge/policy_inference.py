#!/usr/bin/env python3

""" The following commands are used to run the policy inference launcher:
    python3 policy_inference.py --policy /path/to/policy.pt --device cpu"""

import torch
import numpy as np
import argparse
import os
import torch.nn as nn
from typing import Dict

# Define a function to parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Test policy inference for Franka Lift Cube task")
    parser.add_argument("--policy", type=str, required=True, help="Path to the trained policy file (.pt or .jit)")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run inference on (cpu or cuda)")
    return parser.parse_args()

# Define the PolicyLoader class
class PolicyLoader:
    def __init__(self, policy_path: str, device: str = "cpu"):
        """Initialize the policy tester with the path to a trained policy."""
        self.device = device
        self.policy_path = policy_path
        self.policy = self.load_policy(policy_path)
        
        # Define observation dimensions -> these should match the model input
        self.arm_joint_dim = 7  # 7 DoF for Franka arm
        self.finger_joint_dim = 2  # 2 joints for the gripper fingers
        self.joint_dim = self.arm_joint_dim + self.finger_joint_dim  # 9 total joints
        self.obj_pos_dim = 7  # x, y, z + quaternion (4) position of the object 
        self.target_pos_dim = 7  # position (3) + quaternion (4)
        self.action_dim = 8  # 7 joint positions + 1 gripper command "Output of the policy -> Can be then used by controller"
        
        # Default values
        self.last_action = torch.zeros((1, self.action_dim), device=self.device)

    # Load the policy from a file
    def load_policy(self, policy_path: str):
        """Load a trained policy from a file."""
        if not os.path.exists(policy_path):
            raise FileNotFoundError(f"Policy file not found: {policy_path}")
            
        print(f"\nLoading policy from {policy_path}")
        
        # Load the checkpoint
        checkpoint = torch.load(policy_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            print("\nFound model state dict in checkpoint")
            
            # Create an instance of the appropriate model architecture for the policy
            policy = self.create_policy_model()
            
            # Print model architecture for debugging
            print("\nCreated policy model structure:")
            print(policy)
            
            # Load the weights into the model
            try:
                policy.load_state_dict(checkpoint['model_state_dict'])
                print("\nSuccessfully loaded model weights from state dict")
            except Exception as e:
                print(f"Error loading state dict: {e}")
            
            # Set the model to evaluation mode -> Used for inference
            policy.eval()
            return policy
        
        else:
            raise ValueError("Could not find 'model_state_dict' in the checkpoint")

    def create_policy_model(self):
        """Create an instance of the policy model with the right architecture.
        RSL-RL algorithm uses Actor-Critic architecture for the policy model."""

        class ActorCriticPolicy(nn.Module):
            def __init__(self, input_dim, output_dim):
                super(ActorCriticPolicy, self).__init__()

                # Actor network - this is what we use for inference, actor learns mapping from states to actions
                self.actor = nn.Sequential(
                    nn.Linear(input_dim, 256),
                    nn.ELU(alpha=1.0),
                    nn.Linear(256, 128),
                    nn.ELU(alpha=1.0),
                    nn.Linear(128, 64),
                    nn.ELU(alpha=1.0),
                    nn.Linear(64, output_dim)
                )

                # Critic network - not used during inference but needed for model loading
                self.critic = nn.Sequential(
                    nn.Linear(input_dim, 256),
                    nn.ELU(alpha=1.0),
                    nn.Linear(256, 128),
                    nn.ELU(alpha=1.0),
                    nn.Linear(128, 64),
                    nn.ELU(alpha=1.0),
                    nn.Linear(64, 1)
                )

                # Standard deviation parameter
                self.register_buffer('std', torch.ones(output_dim))

            # Forward pass for the model, x is the input tensor of shape (40,1) # Updated comment
            # This is the function that gets called when we do action = policy(obs)
            def forward(self, x):
                # If input is a dict, convert to tensor
                if isinstance(x, dict):
                    # Concatenate all inputs into a single tensor
                    x = torch.cat([
                        x["joint_pos"],
                        x["joint_vel"],
                        x["object_position"],
                        x["object_orientation"], # Added object_orientation here
                        x["target_object_position"],
                        x["actions"]
                    ], dim=1)

                # For inference we only need the actor output
                return self.actor(x)

        # The input size is 40 now (36 original + 4 object orientation)
        input_dim = 40 # Updated input dimension
        # The output is 8 actions (7 joints + 1 gripper command)
        output_dim = 8

        return ActorCriticPolicy(input_dim, output_dim).to(self.device)

    # Create observation for the policy   
    def create_observation(self, 
                               joint_pos: np.ndarray = None, 
                               joint_vel: np.ndarray = None,
                               object_pos: np.ndarray = None,
                               object_orientation: np.ndarray = None,
                               target_pos: np.ndarray = None) -> Dict[str, torch.Tensor]:
        
        """Create a test observation for the policy."""
        # Default values if none provided
        if joint_pos is None:
            # Default joint positions: 7 arm joints + 2 finger joints
            joint_pos = np.zeros(self.joint_dim)
        if joint_vel is None:
            # Default joint velocities: 7 arm joints + 2 finger joints
            joint_vel = np.zeros(self.joint_dim)
        if object_pos is None:
            # Default object position on table
            object_pos = np.array([0.5, 0.3, 0.055])
        if object_orientation is None:
            # Default object orientation (identity quaternion: qw, qx, qy, qz)
            object_orientation = np.array([1.0, 0.0, 0.0, 0.0])
        if target_pos is None:
            # Default target position above table: position (x,y,z) + quaternion (w,x,y,z)
            target_pos = np.array([0.5, 0, 0.1, 1.0, 0, 0, 0])
            
        # Convert to tensors and add batch dimension
        joint_pos_tensor = torch.tensor(joint_pos, dtype=torch.float32, device=self.device).unsqueeze(0)
        joint_vel_tensor = torch.tensor(joint_vel, dtype=torch.float32, device=self.device).unsqueeze(0)
        object_pos_tensor = torch.tensor(object_pos, dtype=torch.float32, device=self.device).unsqueeze(0)
        object_orientation_tensor = torch.tensor(object_orientation, dtype=torch.float32, device=self.device).unsqueeze(0)
        target_pos_tensor = torch.tensor(target_pos, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # Create observation dictionary (for policies that expect dictionary input)
        obs_dict = {
            "joint_pos": joint_pos_tensor,
            "joint_vel": joint_vel_tensor,
            "object_position": object_pos_tensor,
            "object_orientation": object_orientation_tensor,
            "target_object_position": target_pos_tensor,
            "actions": self.last_action
        }
        
        # Create concatenated observation (for policies that expect tensor input)
        obs_concat = torch.cat([
            joint_pos_tensor,
            joint_vel_tensor,
            object_pos_tensor,
            object_orientation_tensor,
            target_pos_tensor,
            self.last_action
        ], dim=1)
        
        # Verify the dimensions match what the model expects
        expected_dim = 40
        actual_dim = obs_concat.shape[1]
        # Print a warning if the dimensions don't match
        if actual_dim != expected_dim:
            print(f"Warning: Model expects input dimension {expected_dim}, but got {actual_dim}")
            print(f"Dimensions: joint_pos={joint_pos_tensor.shape[1]}, joint_vel={joint_vel_tensor.shape[1]}, " +
                   f"object_pos={object_pos_tensor.shape[1]}, object_orientation={object_orientation_tensor.shape[1]}, " +
                   f"target_pos={target_pos_tensor.shape[1]}, actions={self.last_action.shape[1]}")
        
        # Return a dict containing both the dictionary and tensor representations
        # This allows the model to be flexible in terms of input format
        return {
            "dict": obs_dict,
            "tensor": obs_concat
        }
        
    def run_inference(self, observation):
        """Run inference with the policy."""
        with torch.no_grad():
            try:
                # Try using policy forward method (which uses only the actor)
                # Pytorch automatically invokes the forward method when calling the model
                action = self.policy(observation["tensor"])
            
            except Exception as e:
                print(f"Error running inference with tensor input: {e}")
                # Try with dictionary input as a fallback
                try:
                    action = self.policy(observation["dict"])
                except Exception as e2:
                    print(f"Inference failed with both tensor and dict inputs: {e2}")
                    # Return zeros as a last resort fallback
                    return torch.zeros((1, self.action_dim), device=self.device)
    
        # Update last action for next observation
        self.last_action = action.detach().clone()

        return action
        
    def interpret_action(self, action: torch.Tensor) -> Dict:
        """Interpret the action from the policy."""
        # Extract joint positions and gripper command
        action_np = action.cpu().numpy()[0]
        
        # Extract joint positions and gripper command
        joint_positions = action_np[:-1]  # First 7 values
        gripper_command = action_np[-1]   # Last value
        
        # Interpret gripper command
        gripper_state = "OPEN" if gripper_command > 0 else "CLOSE"
        
        return {
            "joint_positions": joint_positions,
            "gripper_command": gripper_command,
            "gripper_state": gripper_state
        }
        
    def interactive_test(self):
        """Run an interactive test session."""
        print("\n=== Franka Lift Cube Policy Inference Tester ===\n")
        print("This tool helps test policy inference for the Franka Lift Cube task.")
        print("You can input observations and see the resulting actions.\n")

        while True:
            print("\n=== Testing Options ===")
            print("1. Test with default values")
            print("2. Test with custom joint positions")
            print("3. Test with custom object position")
            print("4. Test with custom object orientation (quaternion: qw qx qy qz)")
            print("5. Test with custom target position")
            print("6. Test with all custom values")
            print("q. Quit")

            choice = input("\nEnter choice: ")

            if choice == 'q':
                break

            obs = None # Initialize obs

            if choice == '1':
                obs = self.create_observation()
            elif choice == '2':
                print("\nEnter 9 joint position values separated by spaces (-1 to 1 range):")
                print("(7 arm joints + 2 finger joints)")
                try:
                    values = list(map(float, input().split()))
                    if len(values) != 9:
                        print(f"Expected 9 values, got {len(values)}. Using zeros instead.")
                        values = np.zeros(9)
                    obs = self.create_observation(joint_pos=values)
                except ValueError:
                    print("Invalid input. Using default values.")
                    obs = self.create_observation()
            elif choice == '3':
                print("\nEnter 3 object position values (x, y, z) separated by spaces:")
                try:
                    values = list(map(float, input().split()))
                    if len(values) != 3:
                        print(f"Expected 3 values, got {len(values)}. Using defaults instead.")
                        values = np.array([0.5, 0.3, 0.055]) # Updated default to match PolicyRunner
                    obs = self.create_observation(object_pos=values)
                except ValueError:
                    print("Invalid input. Using default values.")
                    obs = self.create_observation()
            elif choice == '4': # Added case for object orientation
                print("\nEnter 4 object orientation values (qw, qx, qy, qz) separated by spaces:")
                try:
                    values = list(map(float, input().split()))
                    if len(values) != 4:
                        print(f"Expected 4 values, got {len(values)}. Using defaults instead.")
                        values = np.array([1.0, 0.0, 0.0, 0.0]) # Default identity quaternion
                    obs = self.create_observation(object_orientation=values)
                except ValueError:
                    print("Invalid input. Using default values.")
                    obs = self.create_observation()
            elif choice == '5': # Renumbered
                print("\nEnter 7 target position values (x, y, z, qw, qx, qy, qz) separated by spaces:")
                try:
                    values = list(map(float, input().split()))
                    if len(values) != 7:
                        print(f"Expected 7 values, got {len(values)}. Using defaults instead.")
                        values = np.array([0.5, 0, 0.1, 1.0, 0, 0, 0]) # Updated default
                    obs = self.create_observation(target_pos=values)
                except ValueError:
                    print("Invalid input. Using default values.")
                    obs = self.create_observation()
            elif choice == '6': # Renumbered
                try:
                    print("\nEnter 9 joint position values:")
                    joint_pos = list(map(float, input().split()))
                    if len(joint_pos) != 9: raise ValueError("Expected 9 joint pos values")

                    print("\nEnter 9 joint velocity values:")
                    joint_vel = list(map(float, input().split()))
                    if len(joint_vel) != 9: raise ValueError("Expected 9 joint vel values")

                    print("\nEnter 3 object position values (x, y, z):")
                    object_pos = list(map(float, input().split()))
                    if len(object_pos) != 3: raise ValueError("Expected 3 object pos values")

                    print("\nEnter 4 object orientation values (qw, qx, qy, qz):") # Added
                    object_orientation = list(map(float, input().split()))
                    if len(object_orientation) != 4: raise ValueError("Expected 4 object orientation values")

                    print("\nEnter 7 target position values (x, y, z, qw, qx, qy, qz):")
                    target_pos = list(map(float, input().split()))
                    if len(target_pos) != 7: raise ValueError("Expected 7 target pos values")

                    obs = self.create_observation(
                        joint_pos=np.array(joint_pos),
                        joint_vel=np.array(joint_vel),
                        object_pos=np.array(object_pos),
                        object_orientation=np.array(object_orientation), # Added
                        target_pos=np.array(target_pos)
                    )
                except ValueError as e:
                    print(f"Invalid input: {e}. Using default values.")
                    obs = self.create_observation()
            else:
                print("Invalid choice. Using default values.")
                obs = self.create_observation()

            if obs is not None: # Check if obs was created
                print("\n=== Input Observations ===")
                for key, value in obs["dict"].items():
                    # Check if value is a tensor before calling .cpu().numpy()
                    if isinstance(value, torch.Tensor):
                         print(f"{key}: {value.cpu().numpy()}")
                    else:
                         print(f"{key}: {value}") # Should not happen with current structure, but safe check

                print("\n=== Running Inference ===")
                try:
                    action = self.run_inference(obs)
                    print(f"Raw action: {action.cpu().numpy()}")

                    interpreted = self.interpret_action(action)
                    print("\n=== Interpreted Action ===")
                    print(f"Joint positions: {interpreted['joint_positions']}")
                    print(f"Gripper command: {interpreted['gripper_command']} ({interpreted['gripper_state']})")
                except Exception as e:
                    print(f"Error running inference: {e}")

                print("\n=== Observation Dimensions ===")
                print(f"Total observation size (tensor): {obs['tensor'].shape[1]}")

def main():
    # Parse command line arguments
    args = parse_args()
    # Create a policy tester
    loader = PolicyLoader(args.policy, args.device)
    loader.interactive_test()

if __name__ == "__main__":
    main()