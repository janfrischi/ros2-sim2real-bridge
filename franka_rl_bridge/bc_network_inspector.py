#!/usr/bin/env python3
"""
Neural Network Inspector for BC Policy Checkpoints
This script provides detailed inspection of saved BC policy models.
"""

import torch
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np

class NetworkInspector:
    """Comprehensive neural network inspection tool"""
    
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        self.checkpoint = None
        self.model_state = None
        
    def load_checkpoint(self):
        """Load the checkpoint file"""
        try:
            print(f"Loading checkpoint from: {self.checkpoint_path}")
            self.checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
            
            if 'model' in self.checkpoint:
                self.model_state = self.checkpoint['model']
            else:
                print("Warning: 'model' key not found in checkpoint")
                self.model_state = self.checkpoint
                
            print("✓ Checkpoint loaded successfully\n")
            
        except Exception as e:
            print(f"✗ Error loading checkpoint: {e}")
            raise
    
    def print_checkpoint_overview(self):
        """Print high-level checkpoint information"""
        print("=" * 80)
        print("CHECKPOINT OVERVIEW")
        print("=" * 80)
        
        print(f"File size: {Path(self.checkpoint_path).stat().st_size / (1024*1024):.2f} MB")
        print(f"Top-level keys: {list(self.checkpoint.keys())}")
        
        # Print metadata if available
        if 'algo_name' in self.checkpoint:
            print(f"Algorithm: {self.checkpoint['algo_name']}")
            
        if 'env_metadata' in self.checkpoint:
            env_meta = self.checkpoint['env_metadata']
            print(f"Environment: {env_meta}")
            
        if 'shape_metadata' in self.checkpoint:
            shape_meta = self.checkpoint['shape_metadata']
            print(f"Shape metadata: {shape_meta}")
            
        if 'config' in self.checkpoint:
            print(f"Config type: {type(self.checkpoint['config'])}")
            if isinstance(self.checkpoint['config'], str):
                try:
                    config = json.loads(self.checkpoint['config'])
                    print("Config preview:")
                    for key, value in list(config.items())[:5]:
                        print(f"  {key}: {value}")
                    if len(config) > 5:
                        print(f"  ... and {len(config) - 5} more config items")
                except:
                    print("  Config is string but not JSON parseable")
        
        print()
    
    def analyze_model_architecture(self):
        """Analyze the neural network architecture"""
        print("=" * 80)
        print("MODEL ARCHITECTURE ANALYSIS")
        print("=" * 80)
        
        if self.model_state is None:
            print("No model state available for analysis")
            return
        
        # Group parameters by component
        components = self.group_parameters_by_component()
        
        print(f"Total parameters: {len(self.model_state)}")
        print(f"Identified components: {len(components)}")
        print()
        
        total_params = 0
        for component_name, params in components.items():
            print(f"📦 {component_name.upper()}")
            print("-" * 40)
            
            component_params = 0
            for param_name, tensor in params.items():
                param_count = tensor.numel()
                component_params += param_count
                print(f"  {param_name:<30} {str(tensor.shape):<20} {param_count:>10,} params")
            
            print(f"  {'Total':<30} {'':<20} {component_params:>10,} params")
            print()
            total_params += component_params
        
        print(f"🔢 TOTAL MODEL PARAMETERS: {total_params:,}")
        print(f"💾 Model size: {total_params * 4 / (1024*1024):.2f} MB (float32)")
        print()
    
    def group_parameters_by_component(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Group parameters by neural network component"""
        components = {}
        
        for param_name, tensor in self.model_state.items():
            # Parse the hierarchical parameter name
            parts = param_name.split('.')
            
            if 'lstm' in param_name.lower():
                component = 'lstm'
            elif 'mean' in param_name.lower():
                component = 'mean_head'
            elif 'scale' in param_name.lower():
                component = 'scale_head'
            elif 'logits' in param_name.lower():
                component = 'logits_head'
            elif 'decoder' in param_name.lower():
                component = 'decoder'
            elif 'encoder' in param_name.lower():
                component = 'encoder'
            elif 'actor' in param_name.lower():
                component = 'actor_layers'
            else:
                component = 'other'
            
            if component not in components:
                components[component] = {}
            
            components[component][param_name] = tensor
        
        return components
    
    def analyze_parameter_statistics(self):
        """Analyze parameter statistics"""
        print("=" * 80)
        print("PARAMETER STATISTICS")
        print("=" * 80)
        
        for param_name, tensor in self.model_state.items():
            if not isinstance(tensor, torch.Tensor):
                continue
                
            stats = {
                'shape': tensor.shape,
                'dtype': tensor.dtype,
                'mean': tensor.mean().item(),
                'std': tensor.std().item(),
                'min': tensor.min().item(),
                'max': tensor.max().item(),
                'zeros': (tensor == 0).sum().item(),
                'total': tensor.numel()
            }
            
            print(f"📊 {param_name}")
            print(f"   Shape: {stats['shape']}")
            print(f"   Type: {stats['dtype']}")
            print(f"   Mean: {stats['mean']:.6f}, Std: {stats['std']:.6f}")
            print(f"   Range: [{stats['min']:.6f}, {stats['max']:.6f}]")
            print(f"   Zeros: {stats['zeros']}/{stats['total']} ({100*stats['zeros']/stats['total']:.1f}%)")
            print()
    
    def analyze_lstm_structure(self):
        """Analyze LSTM-specific structure"""
        print("=" * 80)
        print("LSTM STRUCTURE ANALYSIS")
        print("=" * 80)
        
        lstm_params = {k: v for k, v in self.model_state.items() if 'lstm' in k.lower()}
        
        if not lstm_params:
            print("No LSTM parameters found")
            return
        
        print(f"LSTM parameters found: {len(lstm_params)}")
        
        # Analyze LSTM dimensions
        for param_name, tensor in lstm_params.items():
            print(f"  {param_name}: {tensor.shape}")
            
            # Try to infer LSTM structure from weight shapes
            if 'weight_ih_l0' in param_name:
                input_size = tensor.shape[1]
                hidden_size = tensor.shape[0] // 4  # LSTM has 4 gates
                print(f"    → Input size: {input_size}")
                print(f"    → Hidden size: {hidden_size}")
            elif 'weight_hh_l0' in param_name:
                hidden_size = tensor.shape[1]
                print(f"    → Hidden size: {hidden_size}")
        
        print()
    
    def analyze_gmm_structure(self):
        """Analyze Gaussian Mixture Model structure"""
        print("=" * 80)
        print("GMM STRUCTURE ANALYSIS")
        print("=" * 80)
        
        gmm_components = ['mean', 'scale', 'logits']
        
        for component in gmm_components:
            params = {k: v for k, v in self.model_state.items() if component in k.lower()}
            
            if params:
                print(f"📈 {component.upper()} HEAD:")
                for param_name, tensor in params.items():
                    print(f"  {param_name}: {tensor.shape}")
                    
                    if 'weight' in param_name:
                        input_dim = tensor.shape[1]
                        output_dim = tensor.shape[0]
                        print(f"    → Input dim: {input_dim}")
                        print(f"    → Output dim: {output_dim}")
                        
                        if component in ['mean', 'scale']:
                            # Try to infer num_modes and action_dim
                            action_dim = 8  # Known from your setup
                            num_modes = output_dim // action_dim
                            print(f"    → Inferred modes: {num_modes}, Action dim: {action_dim}")
                        elif component == 'logits':
                            print(f"    → Number of mixture components: {output_dim}")
                
                print()
    
    def export_summary(self, output_file: str = None):
        """Export detailed summary to file"""
        if output_file is None:
            output_file = f"{Path(self.checkpoint_path).stem}_inspection.txt"
        
        # Redirect print to file
        import sys
        original_stdout = sys.stdout
        
        try:
            with open(output_file, 'w') as f:
                sys.stdout = f
                
                self.print_checkpoint_overview()
                self.analyze_model_architecture()
                self.analyze_parameter_statistics()
                self.analyze_lstm_structure()
                self.analyze_gmm_structure()
                
            sys.stdout = original_stdout
            print(f"📄 Detailed inspection saved to: {output_file}")
            
        except Exception as e:
            sys.stdout = original_stdout
            print(f"Error saving inspection: {e}")
    
    def run_full_inspection(self, save_report: bool = True):
        """Run complete inspection"""
        self.load_checkpoint()
        self.print_checkpoint_overview()
        self.analyze_model_architecture()
        self.analyze_lstm_structure()
        self.analyze_gmm_structure()
        
        if save_report:
            self.export_summary()

def main():
    parser = argparse.ArgumentParser(description="Inspect BC Policy Neural Network")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint file")
    parser.add_argument("--stats", action="store_true", help="Include detailed parameter statistics")
    parser.add_argument("--save", action="store_true", help="Save inspection report to file")
    parser.add_argument("--output", type=str, help="Output file name for report")
    
    args = parser.parse_args()
    
    inspector = NetworkInspector(args.checkpoint)
    
    try:
        inspector.load_checkpoint()
        inspector.print_checkpoint_overview()
        inspector.analyze_model_architecture()
        inspector.analyze_lstm_structure()
        inspector.analyze_gmm_structure()
        
        if args.stats:
            inspector.analyze_parameter_statistics()
        
        if args.save:
            inspector.export_summary(args.output)
            
    except Exception as e:
        print(f"Inspection failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())