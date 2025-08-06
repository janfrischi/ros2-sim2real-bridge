#!/usr/bin/env python3
"""
Extract cube configurations from bc_stack_task_test_cases_extended.json 
and generate the testing_configs dictionary for spawn_cubes_testing method.
"""

import json
import numpy as np
from pathlib import Path

def load_test_cases(json_file_path: str):
    """Load test cases from JSON file"""
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    return data['configurations']

def generate_testing_configs_code(configurations):
    """Generate Python code for the testing_configs dictionary"""
    
    print("# Generated testing_configs for spawn_cubes_testing method")
    print("# Copy this into your cube_manager.py file\n")
    print("testing_configs = {")
    
    for i, config in enumerate(configurations):
        config_name = f"config_{i}"
        description = config['description']
        poses = config['poses']
        
        print(f'    "{config_name}": {{')
        print(f'        # {description}')
        
        for j, pose in enumerate(poses):
            cube_name = f"cube_{j+1}"
            pos = pose['pos']
            print(f"        '{cube_name}': np.array([{pos[0]}, {pos[1]}, {pos[2]}]),")
        
        print("    },")
    
    print("}")

def generate_key_mapping(configurations):
    """Generate key mapping for the UI"""
    print("\n# Key mapping for check_keyboard_input_testing method:")
    print("# Add these cases to your keyboard input handler\n")
    
    for i, config in enumerate(configurations):
        if i < 10:  # Use number keys 0-9
            key = str(i)
            config_name = f"config_{i}"
            description = config['description']
            
            print(f"elif key == '{key}':  # {key} - {description}")
            print(f"    self.spawn_cubes_testing(\"{config_name}\")")
            print(f"    self.current_config_name = \"{description}\"")
            print()

def generate_list_configs_info(configurations):
    """Generate the configs_info dictionary for list_testing_configs method"""
    print("\n# configs_info for list_testing_configs method:")
    print("configs_info = {")
    
    for i, config in enumerate(configurations):
        config_name = f"config_{i}"
        description = config['description']
        print(f'    "{config_name}": "{description}",')
    
    print("}")

def print_summary(configurations):
    """Print summary of configurations"""
    print(f"\n📊 SUMMARY")
    print("=" * 50)
    print(f"Total configurations: {len(configurations)}")
    print(f"Keys 0-9 will map to first 10 configurations")
    if len(configurations) > 10:
        print(f"⚠️  {len(configurations) - 10} configurations exceed number keys (0-9)")
        print("   Consider using letter keys or reducing configurations")
    
    print("\n📋 CONFIGURATION LIST:")
    for i, config in enumerate(configurations):
        key = str(i) if i < 10 else "N/A"
        print(f"  [{key}] config_{i}: {config['description']}")

def main():
    """Main function"""
    # Path to the JSON file
    json_file_path = "bc_stack_task_test_cases_extended.json"
    
    # Check if file exists
    if not Path(json_file_path).exists():
        print(f"❌ Error: File {json_file_path} not found!")
        print("Please make sure the file is in the current directory.")
        return
    
    try:
        # Load configurations
        configurations = load_test_cases(json_file_path)
        
        print("🎯 CUBE CONFIGURATION EXTRACTOR")
        print("=" * 60)
        print(f"📁 Source file: {json_file_path}")
        print(f"📊 Found {len(configurations)} configurations")
        print("=" * 60)
        
        # Generate the testing_configs dictionary
        generate_testing_configs_code(configurations)
        
        # Generate key mapping for keyboard input
        generate_key_mapping(configurations)
        
        # Generate configs_info for list method
        generate_list_configs_info(configurations)
        
        # Print summary
        print_summary(configurations)
        
        print("\n✅ Code generation completed!")
        print("📋 Copy the generated code sections into your cube_manager.py file")
        
    except Exception as e:
        print(f"❌ Error processing file: {e}")

if __name__ == "__main__":
    main()