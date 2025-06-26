#!/usr/bin/env python3
"""
Standalone BC Policy Tester
This script allows manual testing of the BC LSTM+GMM policy with custom 48D observation inputs.
"""
import torch
import torch.nn as nn
import numpy as np
import argparse
from typing import Dict, Optional

class LSTMGMMNetwork(nn.Module):
    """LSTM + GMM Actor Network for Behavior Cloning - IsaacLab Compatible"""
    
    def __init__(self, obs_dim: int = 48, action_dim: int = 8, hidden_dim: int = 400, 
                 num_layers: int = 2, num_modes: int = 5, min_std: float = 0.0001,
                 std_activation: str = "softplus", low_noise_eval: bool = True):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_modes = num_modes
        self.min_std = min_std
        self.std_activation = std_activation
        self.low_noise_eval = low_noise_eval
        
        # Architecture for LSTM Network
        self.lstm = nn.LSTM(
            input_size=obs_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )

        # Architecture for Gaussian Mixture Model (GMM) heads
        self.gmm = nn.ModuleDict({
            'mean': nn.Linear(hidden_dim, num_modes * action_dim),  
            'scale': nn.Linear(hidden_dim, num_modes * action_dim),
            'logits': nn.Linear(hidden_dim, num_modes) 
            })

        # Hidden states for sequential inference
        self.hidden_states = None
    
    def reset_hidden_states(self, batch_size: int = 1):
        """Reset LSTM hidden states"""
        device = next(self.parameters()).device
        self.hidden_states = (
            torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device),
            torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        )
    
    def forward(self, obs: Dict[str, torch.Tensor], deterministic: bool = True) -> torch.Tensor:
        """Forward pass through the network"""
        # Concatenate dictionary observations into a single tensor
        obs_tensor = torch.cat([
            obs['eef_pos'],      # [batch_size, seq_length, 3]
            obs['eef_quat'],     # [batch_size, seq_length, 4]  
            obs['gripper_pos'],  # [batch_size, seq_length, 2]
            obs['object']        # [batch_size, seq_length, 39]
        ], dim=-1)  # Result: [batch_size, seq_length, 48]
        
        batch_size, seq_len, _ = obs_tensor.shape
        
        # Initialize hidden states if needed
        if self.hidden_states is None:
            self.reset_hidden_states(batch_size)
            
        # LSTM forward pass
        lstm_out, self.hidden_states = self.lstm(obs_tensor, self.hidden_states)
        
        # Take the last timestep output
        h_t = lstm_out[:, -1, :]  # [batch_size, hidden_dim]
        
        # Pass through GMM heads
        means = self.gmm['mean'](h_t).view(batch_size, self.num_modes, self.action_dim)
        scales = nn.functional.softplus(self.gmm['scale'](h_t)).view(batch_size, self.num_modes, self.action_dim)
        scales = torch.clamp(scales, min=self.min_std)
        logits = self.gmm['logits'](h_t)

        # Apply low noise evaluation if enabled
        if self.low_noise_eval and not self.training:
            scales = scales * 0.1

        # Handle deterministic vs stochastic action selection
        if deterministic:
            # Return mean of most likely mode
            mode_probs = torch.softmax(logits, dim=-1)
            best_mode = torch.argmax(mode_probs, dim=-1)
            actions = means[torch.arange(batch_size), best_mode]
            return actions, means, scales, logits
        else:
            # Stochastic sampling
            mode_probs = torch.softmax(logits, dim=-1)
            mode_dist = torch.distributions.Categorical(mode_probs)
            selected_modes = mode_dist.sample()
            
            selected_means = means[torch.arange(batch_size), selected_modes]
            selected_scales = scales[torch.arange(batch_size), selected_modes]

            gaussian_dist = torch.distributions.Normal(selected_means, selected_scales)
            actions = gaussian_dist.sample()
            return actions, means, scales, logits

class BCPolicyTester:
    """Standalone tester for BC policy"""
    
    def __init__(self, policy_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.policy = self.load_policy(policy_path)
        self.policy.eval()
        
        # Sequence length for RNN
        self.seq_length = 10
        
    def load_policy(self, policy_path: str) -> LSTMGMMNetwork:
        """Load the trained BC policy from checkpoint"""
        try:
            checkpoint = torch.load(policy_path, map_location=self.device)
            state_dict = checkpoint['model']
        
            policy = LSTMGMMNetwork(
                obs_dim=48,
                action_dim=8,
                hidden_dim=400,
                num_layers=2,
                num_modes=5,
                min_std=0.0001,
                std_activation="softplus",
                low_noise_eval=True
            ).to(self.device)
            
            # Map weights from IsaacLab format
            policy_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('policy.nets.rnn.nets.'):
                    new_key = key.replace('policy.nets.rnn.nets.', 'lstm.')
                    policy_state_dict[new_key] = value
                elif key.startswith('policy.nets.rnn.per_step_net.nets.'):
                    new_key = key.replace('policy.nets.rnn.per_step_net.nets.', 'gmm.')
                    policy_state_dict[new_key] = value
                
            policy.load_state_dict(policy_state_dict, strict=True)
            policy.reset_hidden_states(batch_size=1)
            
            print("✅ Successfully loaded BC policy")
            print(f"📊 Model parameters: {sum(p.numel() for p in policy.parameters()):,}")
            return policy
            
        except Exception as e:
            print(f"❌ Error loading policy: {e}")
            raise

    def create_observation_from_vector(self, obs_vector: np.ndarray) -> Dict[str, torch.Tensor]:
        """Create observation dictionary from 48D vector"""
        if len(obs_vector) != 48:
            raise ValueError(f"Observation vector must be 48D, got {len(obs_vector)}D")
        
        # Split the 48D vector into components
        eef_pos = obs_vector[:3]        # [0:3]   - 3D
        eef_quat = obs_vector[3:7]      # [3:7]   - 4D  
        gripper_pos = obs_vector[7:9]   # [7:9]   - 2D
        object_state = obs_vector[9:48] # [9:48]  - 39D
        
        # Convert to tensors with proper dimensions [batch_size=1, seq_length, feature_dim]
        obs_dict = {
            'eef_pos': torch.from_numpy(eef_pos).float().unsqueeze(0).unsqueeze(0).to(self.device),      # [1, 1, 3]
            'eef_quat': torch.from_numpy(eef_quat).float().unsqueeze(0).unsqueeze(0).to(self.device),    # [1, 1, 4]
            'gripper_pos': torch.from_numpy(gripper_pos).float().unsqueeze(0).unsqueeze(0).to(self.device), # [1, 1, 2]
            'object': torch.from_numpy(object_state).float().unsqueeze(0).unsqueeze(0).to(self.device)   # [1, 1, 39]
        }
        
        return obs_dict

    def create_sequence_observation(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Repeat single observation to create sequence of length seq_length"""
        seq_obs = {}
        for key, value in obs_dict.items():
            # Repeat along sequence dimension to create [1, seq_length, feature_dim]
            seq_obs[key] = value.repeat(1, self.seq_length, 1)
        return seq_obs

    def test_inference(self, obs_vector: np.ndarray, deterministic: bool = True, verbose: bool = True):
        """Run inference on a single observation vector"""
        try:
            # Create observation dictionary
            obs_dict = self.create_observation_from_vector(obs_vector)
            seq_obs_dict = self.create_sequence_observation(obs_dict)
            
            if verbose:
                print("\n" + "="*80)
                print("🧠 RUNNING POLICY INFERENCE")
                print("="*80)
                
                # Print input breakdown
                print("📥 INPUT BREAKDOWN:")
                print(f"   EEF Position:    [{obs_vector[0]:+7.4f}, {obs_vector[1]:+7.4f}, {obs_vector[2]:+7.4f}]")
                print(f"   EEF Quaternion:  [{obs_vector[3]:+7.4f}, {obs_vector[4]:+7.4f}, {obs_vector[5]:+7.4f}, {obs_vector[6]:+7.4f}] (qw,qx,qy,qz)")
                print(f"   Gripper Pos:     [{obs_vector[7]:+7.4f}, {obs_vector[8]:+7.4f}]")
                print(f"   Object State:    {obs_vector[9:48]} (39D)")
                print("-"*80)
            
            # Run inference
            with torch.no_grad():
                if deterministic:
                    action, means, scales, logits = self.policy(seq_obs_dict, deterministic=True)
                else:
                    action, means, scales, logits = self.policy(seq_obs_dict, deterministic=False)
            
            # Convert to numpy
            action_np = action.cpu().numpy().squeeze()
            means_np = means.cpu().numpy().squeeze()  # [num_modes, action_dim]
            scales_np = scales.cpu().numpy().squeeze()  # [num_modes, action_dim]
            logits_np = logits.cpu().numpy().squeeze()  # [num_modes]
            mode_probs = torch.softmax(logits, dim=-1).cpu().numpy().squeeze()
            
            if verbose:
                # Print GMM analysis
                print("🎯 GMM ANALYSIS:")
                print(f"   Mode Probabilities: {mode_probs}")
                print(f"   Best Mode:          {np.argmax(mode_probs)} (prob: {np.max(mode_probs):.4f})")
                print("-"*80)
                
                # Print all mode means and scales
                print("📊 ALL MODES:")
                for i in range(len(mode_probs)):
                    print(f"   Mode {i} (prob: {mode_probs[i]:.4f}):")
                    print(f"     Mean:  [{', '.join([f'{x:+7.4f}' for x in means_np[i]])}]")
                    print(f"     Scale: [{', '.join([f'{x:+7.4f}' for x in scales_np[i]])}]")
                print("-"*80)
                
                # Print final action
                mode_type = "DETERMINISTIC" if deterministic else "STOCHASTIC"
                print(f"🎬 FINAL ACTION ({mode_type}):")
                print(f"   Full Action: [{', '.join([f'{x:+7.4f}' for x in action_np])}]")
                print(f"   EEF Pose:    [{', '.join([f'{x:+7.4f}' for x in action_np[:7]])}] (x,y,z,qw,qx,qy,qz)")
                print(f"   Gripper:     {action_np[7]:+7.4f}")
                print("="*80)
            
            return {
                'action': action_np,
                'means': means_np,
                'scales': scales_np,
                'logits': logits_np,
                'mode_probs': mode_probs,
                'best_mode': np.argmax(mode_probs)
            }
            
        except Exception as e:
            print(f"❌ Error during inference: {e}")
            import traceback
            traceback.print_exc()
            return None

    def interactive_test(self):
        """Interactive testing mode"""
        print("\n" + "="*80)
        print("🔬 BC POLICY INTERACTIVE TESTER")
        print("="*80)
        print("Commands:")
        print("  'zeros' - Test with 48D zero vector")
        print("  'random' - Test with random 48D vector")
        print("  'custom' - Input custom 48D vector")
        print("  'reset' - Reset LSTM hidden states")
        print("  'both' - Run both deterministic and stochastic inference")
        print("  'quit' - Exit")
        print("="*80)
        
        while True:
            try:
                command = input("\n🎮 Enter command: ").strip().lower()
                
                if command == 'quit':
                    break
                elif command == 'reset':
                    self.policy.reset_hidden_states(batch_size=1)
                    print("🔄 LSTM hidden states reset")
                elif command == 'zeros':
                    obs_vector = np.zeros(48)
                    print("🔢 Testing with zero vector...")
                    self.test_inference(obs_vector, deterministic=True)
                elif command == 'random':
                    obs_vector = np.random.randn(48) * 0.1  # Small random values
                    print("🎲 Testing with random vector...")
                    self.test_inference(obs_vector, deterministic=True)
                elif command == 'custom':
                    try:
                        print("📝 Enter 48D observation vector (space-separated):")
                        user_input = input("   > ")
                        values = [float(x) for x in user_input.split()]
                        if len(values) != 48:
                            print(f"❌ Expected 48 values, got {len(values)}")
                            continue
                        obs_vector = np.array(values)
                        self.test_inference(obs_vector, deterministic=True)
                    except ValueError:
                        print("❌ Invalid input. Please enter 48 space-separated numbers.")
                elif command == 'both':
                    obs_vector = np.random.randn(48) * 0.1
                    print("🎲 Testing DETERMINISTIC inference:")
                    result_det = self.test_inference(obs_vector, deterministic=True, verbose=False)
                    print("\n🎲 Testing STOCHASTIC inference:")
                    result_stoch = self.test_inference(obs_vector, deterministic=False, verbose=False)
                    
                    if result_det and result_stoch:
                        print("\n📊 COMPARISON:")
                        print(f"   Deterministic: [{', '.join([f'{x:+7.4f}' for x in result_det['action']])}]")
                        print(f"   Stochastic:    [{', '.join([f'{x:+7.4f}' for x in result_stoch['action']])}]")
                        diff = np.abs(result_det['action'] - result_stoch['action'])
                        print(f"   Abs Difference: [{', '.join([f'{x:+7.4f}' for x in diff])}]")
                        print(f"   Max Difference: {np.max(diff):+7.4f}")
                else:
                    print("❌ Unknown command. Try 'zeros', 'random', 'custom', 'reset', 'both', or 'quit'")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="BC Policy Standalone Tester")
    parser.add_argument("--policy", type=str, required=True,
                       help="Path to the trained BC policy file (.pt)")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to run inference on (cpu or cuda)")
    parser.add_argument("--obs", type=str, default=None,
                       help="48D observation vector (space-separated) for single test")
    parser.add_argument("--deterministic", action="store_true",
                       help="Use deterministic inference")
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive mode")
    
    args = parser.parse_args()
    
    try:
        # Create tester
        tester = BCPolicyTester(args.policy, args.device)
        
        if args.interactive:
            # Interactive mode
            tester.interactive_test()
        elif args.obs:
            # Single observation test
            try:
                values = [float(x) for x in args.obs.split()]
                if len(values) != 48:
                    print(f"❌ Expected 48 values, got {len(values)}")
                    return
                obs_vector = np.array(values)
                tester.test_inference(obs_vector, deterministic=args.deterministic)
            except ValueError:
                print("❌ Invalid observation format. Please provide 48 space-separated numbers.")
        else:
            # Default: test with zeros
            print("🔢 Testing with 48D zero vector (use --interactive for more options)...")
            obs_vector = np.zeros(48)
            tester.test_inference(obs_vector, deterministic=args.deterministic)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()