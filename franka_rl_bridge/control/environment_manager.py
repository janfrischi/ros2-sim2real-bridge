#!/usr/bin/env python3
"""
Environment Manager - Handles cube spawning and environment state
"""
import numpy as np
import random
from typing import Dict

class EnvironmentManager:
    """Manages environment state and cube spawning patterns"""
    
    def __init__(self):
        self.cube_spawn_config = {
            "pose_range": {
                "x": (0.4, 0.6),
                "y": (-0.3, 0.3),
                "z": (0.0203, 0.0203)
            },
            "min_cube_distance": 0.05
        }
        
        self.color_names = {
            'cube_1': 'Blue Cube  ',
            'cube_2': 'Red Cube   ',
            'cube_3': 'Green Cube '
        }
    
    def randomly_spawn_cubes(self) -> Dict[str, np.ndarray]:
        """Randomly spawn cubes within specified range"""
        print("\n🎲 RANDOMLY SPAWNING CUBES")
        print("=" * 50)
        
        pose_range = self.cube_spawn_config["pose_range"]
        min_distance = self.cube_spawn_config["min_cube_distance"]
        
        new_positions = {}
        cube_names = ['cube_1', 'cube_2', 'cube_3']
        
        for i, cube_name in enumerate(cube_names):
            max_attempts = 50
            attempts = 0
            
            while attempts < max_attempts:
                # Generate random position
                x = random.uniform(pose_range["x"][0], pose_range["x"][1])
                y = random.uniform(pose_range["y"][0], pose_range["y"][1])
                z = pose_range["z"][0]
                
                new_pos = np.array([x, y, z])
                
                # Check distance from existing cubes
                valid_position = True
                for existing_cube, existing_pos in new_positions.items():
                    distance = np.linalg.norm(new_pos - existing_pos)
                    if distance < min_distance:
                        valid_position = False
                        break
                
                if valid_position:
                    new_positions[cube_name] = new_pos
                    break
                
                attempts += 1
            
            if attempts >= max_attempts:
                # Fallback positions
                fallback_positions = {
                    'cube_1': np.array([0.45, -0.15, 0.0203]),
                    'cube_2': np.array([0.5, 0.0, 0.0203]),
                    'cube_3': np.array([0.55, 0.15, 0.0203])
                }
                new_positions[cube_name] = fallback_positions[cube_name]
                print(f"⚠️ Using fallback position for {cube_name}")
        
        self._display_positions(new_positions)
        print("✅ Random cube spawning completed!")
        print("=" * 50)
        
        return new_positions
    
    def spawn_cubes_in_pattern(self, pattern: str) -> Dict[str, np.ndarray]:
        """Spawn cubes in predefined patterns"""
        print(f"\n📐 SPAWNING CUBES IN {pattern.upper()} PATTERN")
        print("=" * 50)
        
        if pattern == "line":
            positions = {
                'cube_1': np.array([0.5, -0.2, 0.0203]),
                'cube_2': np.array([0.5, 0.0, 0.0203]),
                'cube_3': np.array([0.5, 0.2, 0.0203])
            }
        elif pattern == "triangle":
            positions = {
                'cube_1': np.array([0.45, -0.1, 0.0203]),
                'cube_2': np.array([0.45, 0.1, 0.0203]),
                'cube_3': np.array([0.55, 0.0, 0.0203])
            }
        elif pattern == "stack_ready":
            positions = {
                'cube_1': np.array([0.5, 0.0, 0.0203]),
                'cube_2': np.array([0.45, -0.15, 0.0203]),
                'cube_3': np.array([0.55, 0.15, 0.0203])
            }
        else:
            print(f"❌ Unknown pattern: {pattern}")
            return {}
        
        self._display_positions(positions)
        print(f"✅ {pattern.capitalize()} pattern applied!")
        print("=" * 50)
        
        return positions
    
    def _display_positions(self, positions: Dict[str, np.ndarray]):
        """Display cube positions"""
        print("📍 NEW CUBE POSITIONS:")
        for cube_name, pos in positions.items():
            print(f"  {self.color_names[cube_name]}: [{pos[0]:+7.4f}, {pos[1]:+7.4f}, {pos[2]:+7.4f}]")