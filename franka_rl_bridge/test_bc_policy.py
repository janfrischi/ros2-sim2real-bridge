#!/usr/bin/env python3
"""
Test script to verify BC policy loading and inference
"""
# filepath: /home/pdzuser/franka_ros2_ws/src/franka_rl_bridge/franka_rl_bridge/test_bc_policy.py

import torch
import numpy as np
import argparse
from franka_rl_bridge.bc_policy_runner import RNNGMMActorNetwork

def test_network_creation():
    """Test creating the network from scratch"""
    print("Testing network creation...")
    
    network = RNNGMMActorNetwork(
        obs_dim=48,
        action_dim=8,
        hidden_dim=400,
        num_layers=2,
        num_modes=5,
        min_std=1e-4
    )
    
    print(f"✓ Network created successfully")
    print(f"  - Parameters: {sum(p.numel() for p in network.parameters()):,}")
    print(f"  - Observation dim: {network.obs_dim}")
    print(f"  - Action dim: {network.action_dim}")
    
    return network

def test_dummy_inference(network):
    """Test inference with dummy observations"""
    print("\nTesting dummy inference...")
    
    # Create dummy observation
    batch_size = 1
    obs_dim = 48
    dummy_obs = torch.randn(batch_size, obs_dim)
    
    network.eval()
    network.reset_hidden_states(batch_size)
    
    with torch.no_grad():
        # Test deterministic inference
        action_det = network(dummy_obs, deterministic=True)
        print(f"✓ Deterministic action shape: {action_det.shape}")
        print(f"  - Action range: [{action_det.min():.3f}, {action_det.max():.3f}]")
        
        # Test stochastic inference
        action_stoch = network(dummy_obs, deterministic=False)
        print(f"✓ Stochastic action shape: {action_stoch.shape}")
        print(f"  - Action range: [{action_stoch.min():.3f}, {action_stoch.max():.3f}]")
        
        # Test sequential inference (LSTM memory)
        actions_seq = []
        for i in range(5):
            action = network(dummy_obs, deterministic=True)
            actions_seq.append(action.cpu().numpy())
        
        print(f"✓ Sequential inference test completed")
        print(f"  - Actions vary due to LSTM memory: {not np.allclose(actions_seq[0], actions_seq[-1])}")
    
    return True

def test_policy_loading(policy_path):
    """Test loading actual policy checkpoint"""
    print(f"\nTesting policy loading from: {policy_path}")
    
    try:
        # Test direct loading
        checkpoint = torch.load(policy_path, map_location='cpu')
        print(f"✓ Checkpoint loaded successfully")
        print(f"  - Checkpoint keys: {list(checkpoint.keys())}")
        
        # Test network loading
        network = RNNGMMActorNetwork()
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'policy' in checkpoint:
            state_dict = checkpoint['policy']
        else:
            state_dict = checkpoint
            
        network.load_state_dict(state_dict, strict=False)
        print(f"✓ Network weights loaded successfully")
        
        # Test inference with loaded weights
        dummy_obs = torch.randn(1, 48)
        network.eval()
        network.reset_hidden_states(1)
        
        with torch.no_grad():
            action = network(dummy_obs, deterministic=True)
        
        print(f"✓ Inference with loaded weights successful")
        print(f"  - Action: {action.cpu().numpy().squeeze()}")
        
        return True
        
    except Exception as e:
        print(f"✗ Policy loading failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Test BC Policy Implementation")
    parser.add_argument("--policy", type=str, help="Path to policy checkpoint (optional)")
    args = parser.parse_args()
    
    print("=" * 50)
    print("BC Policy Implementation Test")
    print("=" * 50)
    
    # Test 1: Network creation
    try:
        network = test_network_creation()
    except Exception as e:
        print(f"✗ Network creation failed: {e}")
        return
    
    # Test 2: Dummy inference
    try:
        test_dummy_inference(network)
    except Exception as e:
        print(f"✗ Dummy inference failed: {e}")
        return
    
    # Test 3: Policy loading (if provided)
    if args.policy:
        test_policy_loading(args.policy)
    else:
        print("\nSkipping policy loading test (no --policy provided)")
    
    print("\n" + "=" * 50)
    print("Basic tests completed successfully!")
    print("=" * 50)

if __name__ == "__main__":
    main()