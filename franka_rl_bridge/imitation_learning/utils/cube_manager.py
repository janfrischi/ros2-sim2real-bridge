import numpy as np
import random

class CubeManagerMixin: 

    def randomly_spawn_cubes(self):
        """Randomly spawn the three cubes within the specified range"""
        print("\n🎲 RANDOMLY SPAWNING CUBES")
        print("=" * 50)
        
        pose_range = self.cube_spawn_config["pose_range"]
        min_distance = self.cube_spawn_config["min_cube_distance"]
        
        new_positions = {}
        cube_names = ['cube_1', 'cube_2', 'cube_3']
        
        for i, cube_name in enumerate(cube_names):
            max_attempts = 50  # Prevent infinite loop
            attempts = 0
            
            while attempts < max_attempts:
                # Generate random position
                x = random.uniform(pose_range["x"][0], pose_range["x"][1])
                y = random.uniform(pose_range["y"][0], pose_range["y"][1])
                z = pose_range["z"][0]  # Fixed Z value (table height)
                
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
                # Fallback: use a safe position if we can't find a valid random one
                fallback_positions = {
                    'cube_1': np.array([0.45, -0.15, 0.0203]),
                    'cube_2': np.array([0.5, 0.0, 0.0203]),
                    'cube_3': np.array([0.55, 0.15, 0.0203])
                }
                new_positions[cube_name] = fallback_positions[cube_name]
                print(f"⚠️ Using fallback position for {cube_name} after {max_attempts} attempts")
        
    
        self.cube_positions.update(new_positions)
        
        # Display the changes
        print("📍 NEW CUBE POSITIONS:")
        color_names = {
            'cube_1': 'Blue Cube  ',
            'cube_2': 'Red Cube   ',
            'cube_3': 'Green Cube '
        }
        for cube_name, pos in new_positions.items():
            print(f"  {color_names[cube_name]}: [{pos[0]:+7.4f}, {pos[1]:+7.4f}, {pos[2]:+7.4f}]")
            
        
        print("=" * 50)
        print("✅ Cube spawning completed!")
        
        # Reset policy episode state since environment changed
        if hasattr(self, 'policy') and self.policy:
            self.policy.start_episode()
            print("🔄 Policy episode state reset due to environment change")

    def spawn_cubes_preset(self, preset_name: str = "default"):
        """Spawn cubes using predefined pose presets"""
        print(f"\n🎯 SPAWNING CUBES - PRESET: {preset_name.upper()}")
        print("=" * 50)
        
        # Define all preset configurations
        presets = {
            "default": {
                'cube_1': np.array([0.4221598207950592, -0.1940348893404007, 0.0203]),
                'cube_2': np.array([0.47585567831993103, -0.046219781041145325, 0.0203]),
                'cube_3': np.array([0.4306733310222626, -0.2792506217956543, 0.0203])
            },
            "custom_1": {
                'cube_1': np.array([0.45, -0.10, 0.0203]),
                'cube_2': np.array([0.55, -0.10, 0.0203]),
                'cube_3': np.array([0.45, 0.10, 0.0203])
            },
            "wide_spread": {
                'cube_1': np.array([0.35, -0.25, 0.0203]),
                'cube_2': np.array([0.65, 0.0, 0.0203]),
                'cube_3': np.array([0.50, 0.15, 0.0203])
            },
            "tight_cluster": {
                'cube_1': np.array([0.50, -0.1, 0.0203]),
                'cube_2': np.array([0.50, 0.0, 0.0203]),
                'cube_3': np.array([0.50, 0.1, 0.0203])
            },
            "corner_formation": {
                'cube_1': np.array([0.40, -0.20, 0.0203]),  # Bottom left
                'cube_2': np.array([0.40, 0.20, 0.0203]),   # Top left
                'cube_3': np.array([0.60, 0.0, 0.0203])     # Right center
            },
            "stacking_ready": {
                'cube_1': np.array([0.50, 0.0, 0.0203]),    # Target base
                'cube_2': np.array([0.40, -0.15, 0.0203]),  # Source 1
                'cube_3': np.array([0.60, 0.15, 0.0203])    # Source 2
            },
            "manipulation_test": {
                'cube_1': np.array([0.45, -0.10, 0.0203]),
                'cube_2': np.array([0.55, 0.10, 0.0203]),
                'cube_3': np.array([0.50, 0.0, 0.0203])
            },
            "reach_challenge": {
                'cube_1': np.array([0.50, -0.30, 0.0203]),  # Far left
                'cube_2': np.array([0.50, 0.30, 0.0203]),   # Far right
                'cube_3': np.array([0.65, 0.0, 0.0203])     # Center
            },
            "pick_place_demo": {
                'cube_1': np.array([0.40, 0.0, 0.0203]),  # Pick source
                'cube_2': np.array([0.50, 0.10, 0.0203]),   # Place target area
                'cube_3': np.array([0.40, -0.20, 0.0203])   # Obstacle/intermediate
            },
            "sorting_task": {
                'cube_1': np.array([0.38, -0.25, 0.0203]),  # Left bin
                'cube_2': np.array([0.50, 0.0, 0.0203]),    # Center (to sort)
                'cube_3': np.array([0.62, 0.25, 0.0203])    # Right bin
            },
            "assembly_line": {
                'cube_1': np.array([0.35, 0.0, 0.0203]),    # Input
                'cube_2': np.array([0.50, 0.0, 0.0203]),    # Processing
                'cube_3': np.array([0.60, 0.0, 0.0203])     # Output
            },
            "circular_arrangement": {
                'cube_1': np.array([0.50, -0.12, 0.0203]),  # Bottom
                'cube_2': np.array([0.44, 0.06, 0.0203]),   # Top left
                'cube_3': np.array([0.56, 0.06, 0.0203])    # Top right
            },
            "precision_test": {
                'cube_1': np.array([0.48, -0.05, 0.0203]),
                'cube_2': np.array([0.50, 0.0, 0.0203]),
                'cube_3': np.array([0.52, 0.05, 0.0203])
            },
            "learning_progression_1": {
                'cube_1': np.array([0.45, -0.15, 0.0203]),  # Easy reach
                'cube_2': np.array([0.50, 0.0, 0.0203]),    # Medium
                'cube_3': np.array([0.55, 0.15, 0.0203])    # Harder reach
            },
            "learning_progression_2": {
                'cube_1': np.array([0.40, -0.20, 0.0203]),  # Further challenge
                'cube_2': np.array([0.60, 0.20, 0.0203]),   # Cross workspace
                'cube_3': np.array([0.50, 0.0, 0.0203])     # Central reference
            },
            "workspace_corners": {
                'cube_1': np.array([0.35, -0.30, 0.0203]),  # Bottom left corner
                'cube_2': np.array([0.35, 0.30, 0.0203]),   # Top left corner
                'cube_3': np.array([0.65, 0.0, 0.0203])     # Right edge
            }
        }
    
        # Check if preset exists
        if preset_name not in presets:
            available_presets = list(presets.keys())
            print(f"❌ Unknown preset: {preset_name}")
            print(f"📋 Available presets: {', '.join(available_presets)}")
            return
        
        # Get the preset positions
        positions = presets[preset_name]
        
        # Update cube positions
        self.cube_positions.update(positions)
        
        # Reset orientations to identity for all presets
        for cube_name in ['cube_1', 'cube_2', 'cube_3']:
            self.cube_quaternions[cube_name] = np.array([0.0, 0.0, 0.0, 1.0])
        
        # Display new positions with enhanced formatting
        color_names = {
            'cube_1': '🔵 Blue Cube ',
            'cube_2': '🔴 Red Cube  ',
            'cube_3': '🟢 Green Cube'
        }
        
        print("📍 NEW CUBE POSITIONS:")
        for cube_name, pos in positions.items():
            print(f"  {color_names[cube_name]}: [{pos[0]:+7.4f}, {pos[1]:+7.4f}, {pos[2]:+7.4f}]")
        
        # Calculate workspace metrics
        distances = []
        cube_positions_list = list(positions.values())
        for i in range(len(cube_positions_list)):
            for j in range(i+1, len(cube_positions_list)):
                dist = np.linalg.norm(cube_positions_list[i] - cube_positions_list[j])
                distances.append(dist)
        
        min_distance = min(distances)
        max_distance = max(distances)
        avg_distance = np.mean(distances)
        
        print(f"\n📊 WORKSPACE METRICS:")
        print(f"   Min distance between cubes: {min_distance:.4f}m")
        print(f"   Max distance between cubes: {max_distance:.4f}m")
        print(f"   Avg distance between cubes: {avg_distance:.4f}m")
        
        print(f"\n✅ Preset '{preset_name}' applied successfully!")
        print("=" * 50)
        
        # Reset policy episode state
        if hasattr(self, 'policy') and self.policy:
            self.policy.start_episode()
            print("🔄 Policy episode state reset due to environment change")

    def spawn_cubes_testing(self, config_name: str = "config_0"):
        """Spawn cubes using testing configurations from JSON files"""
        print(f"\n🧪 SPAWNING CUBES - TESTING CONFIG: {config_name.upper()}")
        print("=" * 50)
        
        # Define testing configurations (based on bc_stack_task_test_cases_extended.json)
        testing_configs = {
        "config_0": {
            # Line, centered
            'cube_1': np.array([0.45, 0.0, 0.0203]),
            'cube_2': np.array([0.45, 0.2, 0.0203]),
            'cube_3': np.array([0.45, -0.2, 0.0203]),
        }, 
        "config_1": {
            # Spread triangle
            'cube_1': np.array([0.4, -0.2, 0.0203]),
            'cube_2': np.array([0.6, -0.2, 0.0203]),
            'cube_3': np.array([0.5, 0.2, 0.0203]),
        },
        "config_2": {
            # Vertical line
            'cube_1': np.array([0.6, 0.0, 0.0203]),
            'cube_2': np.array([0.5, 0.0, 0.0203]),
            'cube_3': np.array([0.4, 0.0, 0.0203]),
        },
        "config_3": {
            # Random triangle
            'cube_1': np.array([0.4, 0.2, 0.0203]),
            'cube_2': np.array([0.45, -0.1, 0.0203]),
            'cube_3': np.array([0.6, 0.1, 0.0203]),
        },
        "config_4": {
            # Long diagonals
            'cube_1': np.array([0.6, 0.0, 0.0203]),
            'cube_2': np.array([0.5, 0.2, 0.0203]),
            'cube_3': np.array([0.4, -0.2, 0.0203]),
        },
        "config_5": {
            # Line with offset
            'cube_1': np.array([0.42, -0.1, 0.0203]),
            'cube_2': np.array([0.5, 0.0, 0.0203]),
            'cube_3': np.array([0.58, 0.1, 0.0203]),
        },
        "config_6": {
            # Close cluster
            'cube_1': np.array([0.5, 0.1, 0.0203]),
            'cube_2': np.array([0.5, 0.0, 0.0203]),
            'cube_3': np.array([0.5, -0.1, 0.0203]),
        },
        "config_7": {
            # Far y-range
            'cube_1': np.array([0.5, 0.3, 0.0203]),
            'cube_2': np.array([0.5, -0.3, 0.0203]),
            'cube_3': np.array([0.4, 0.0, 0.0203]),
        },
        "config_8": {
            # Edge-to-edge test
            'cube_1': np.array([0.4, 0.25, 0.0203]),
            'cube_2': np.array([0.6, 0.25, 0.0203]),
            'cube_3': np.array([0.5, -0.25, 0.0203]),
        },
        "config_9": {
            # Increasing height
            'cube_1': np.array([0.45, -0.2, 0.0203]),
            'cube_2': np.array([0.5, -0.1, 0.0203]),
            'cube_3': np.array([0.55, 0.0, 0.0203]),
        },
        "config_10": {
            # Centered close
            'cube_1': np.array([0.45, -0.05, 0.0203]),
            'cube_2': np.array([0.5, 0.0, 0.0203]),
            'cube_3': np.array([0.55, 0.05, 0.0203]),
        },
        "config_11": {
            # Side-to-side
            'cube_1': np.array([0.6, -0.2, 0.0203]),
            'cube_2': np.array([0.6, 0.2, 0.0203]),
            'cube_3': np.array([0.4, 0.0, 0.0203]),
        },
        "config_12": {
            # Opposite corners
            'cube_1': np.array([0.5, 0.0, 0.0203]),
            'cube_2': np.array([0.6, 0.25, 0.0203]),
            'cube_3': np.array([0.4, -0.25, 0.0203]),
        },
        "config_13": {
            # Full spread triangle
            'cube_1': np.array([0.4, -0.25, 0.0203]),
            'cube_2': np.array([0.4, 0.25, 0.0203]),
            'cube_3': np.array([0.55, 0.0, 0.0203]),
        },
        "config_14": {
            # Short diagonal pattern in bottom-right
            'cube_1': np.array([0.55, -0.2, 0.0203]),
            'cube_2': np.array([0.5, -0.15, 0.0203]),
            'cube_3': np.array([0.45, -0.1, 0.0203]),
        },
        "config_15": {
            # Inverted triangle in top-center
            'cube_1': np.array([0.5, 0.25, 0.0203]),
            'cube_2': np.array([0.45, 0.1, 0.0203]),
            'cube_3': np.array([0.55, 0.1, 0.0203]),
        },
    }
    
        # Check if config exists
        if config_name not in testing_configs:
            available_configs = list(testing_configs.keys())
            print(f"❌ Unknown testing config: {config_name}")
            print(f"🧪 Available testing configs: {', '.join(available_configs)}")
            return
        
        # Get the config positions
        positions = testing_configs[config_name]
        
        # Update cube positions
        self.cube_positions.update(positions)
        
        # Reset orientations to identity for all configs
        for cube_name in ['cube_1', 'cube_2', 'cube_3']:
            self.cube_quaternions[cube_name] = np.array([0.0, 0.0, 0.0, 1.0])
        
        # Display new positions with enhanced formatting
        color_names = {
            'cube_1': '🔵 Blue Cube ',
            'cube_2': '🔴 Red Cube  ',
            'cube_3': '🟢 Green Cube'
        }
        
        print("📍 NEW CUBE POSITIONS:")
        for cube_name, pos in positions.items():
            print(f"  {color_names[cube_name]}: [{pos[0]:+7.4f}, {pos[1]:+7.4f}, {pos[2]:+7.4f}]")
        
        # Calculate workspace metrics
        distances = []
        cube_positions_list = list(positions.values())
        for i in range(len(cube_positions_list)):
            for j in range(i+1, len(cube_positions_list)):
                dist = np.linalg.norm(cube_positions_list[i] - cube_positions_list[j])
                distances.append(dist)
        
        min_distance = min(distances)
        max_distance = max(distances)
        avg_distance = np.mean(distances)
        
        print(f"\n📊 WORKSPACE METRICS:")
        print(f"   Min distance between cubes: {min_distance:.4f}m")
        print(f"   Max distance between cubes: {max_distance:.4f}m")
        print(f"   Avg distance between cubes: {avg_distance:.4f}m")
        
        print(f"\n✅ Testing config '{config_name}' applied successfully!")
        print("=" * 50)
        
        # Reset policy episode state
        if hasattr(self, 'policy') and self.policy:
            self.policy.start_episode()
            print("🔄 Policy episode state reset due to environment change")

    def list_cube_presets(self):
        """List all available cube presets with descriptions"""
        presets_info = {
            "default": "Original IsaacLab training positions",
            "custom_1": "Your requested custom positions",
            "wide_spread": "Cubes spread across full workspace",
            "tight_cluster": "Cubes close together in center",
            "corner_formation": "L-shaped corner arrangement",
            "stacking_ready": "Optimal positions for stacking tasks",
            "manipulation_test": "Standard manipulation testing layout",
            "reach_challenge": "Tests maximum reach capabilities",
            "pick_place_demo": "Demonstration of pick-and-place",
            "sorting_task": "Three-bin sorting scenario",
            "assembly_line": "Linear assembly sequence",
            "circular_arrangement": "Triangular/circular formation",
            "precision_test": "Close spacing for precision testing",
            "learning_progression_1": "Beginner difficulty progression",
            "learning_progression_2": "Advanced difficulty progression",
            "workspace_corners": "Extreme workspace positions"
        }
        
        print("\n📋 AVAILABLE CUBE PRESETS")
        print("=" * 60)
        for preset, description in presets_info.items():
            print(f"  {preset:<22} │ {description}")
        print("=" * 60)
        print("Usage: Press the corresponding number key or use 'p' + preset name")
        print()

    def spawn_cubes_in_pattern(self, pattern: str = "line"):
        """Spawn cubes in predefined patterns"""
        print(f"\n📐 SPAWNING CUBES IN {pattern.upper()} PATTERN")
        print("=" * 50)
        
        if pattern == "line":
            # Cubes in a line from left to right
            positions = {
                'cube_1': np.array([0.45, -0.2, 0.0203]),
                'cube_2': np.array([0.45, 0.0, 0.0203]),
                'cube_3': np.array([0.45, 0.2, 0.0203])
            }
        elif pattern == "triangle": # this is working
            # Cubes in a triangle formation
            positions = {
                'cube_1': np.array([0.45, -0.1, 0.0203]),
                'cube_2': np.array([0.45, 0.1, 0.0203]),
                'cube_3': np.array([0.55, 0.0, 0.0203])
            }
        elif pattern == "stack_ready": # this is working
            # Cubes positioned for easy stacking
            positions = {
                'cube_1': np.array([0.5, 0.2, 0.0203]),      # Bottom (target)
                'cube_2': np.array([0.5, 0.0, 0.0203]),   # Source 1
                'cube_3': np.array([0.4, -0.2, 0.0203])     # Source 2
            }
        else:
            print(f"❌ Unknown pattern: {pattern}")
            return
        
        # Update positions
        self.cube_positions.update(positions)
        
        # Reset orientations to identity
        for cube_name in ['cube_1', 'cube_2', 'cube_3']:
            self.cube_quaternions[cube_name] = np.array([0.0, 0.0, 0.0, 1.0])
        
        # Display new positions
        color_names = {
            'cube_1': 'Blue Cube  ',
            'cube_2': 'Red Cube   ',
            'cube_3': 'Green Cube '
        }
        
        print("📍 NEW CUBE POSITIONS:")
        for cube_name, pos in positions.items():
            print(f"  {color_names[cube_name]}: [{pos[0]:+7.4f}, {pos[1]:+7.4f}, {pos[2]:+7.4f}]")
        
        print(f"✅ {pattern.capitalize()} pattern applied!")
        print("=" * 50)
        
        # Reset policy episode state
        if hasattr(self, 'policy') and self.policy:
            self.policy.start_episode()
            print("🔄 Policy episode state reset due to environment change")


    def update_attached_cube_pose(self):
        """
        Update the position and orientation of the attached cube to match the end-effector
        """

        # Get current end effector pose
        eef_pos = np.array([
            self.current_eef_pose.pose.position.x,
            self.current_eef_pose.pose.position.y,
            self.current_eef_pose.pose.position.z
        ])

        # Get current end effector quaternion
        eef_quat_ros = np.array([
            self.current_eef_pose.pose.orientation.x,
            self.current_eef_pose.pose.orientation.y,
            self.current_eef_pose.pose.orientation.z,
            self.current_eef_pose.pose.orientation.w
        ])

        # Convert to the IsaacLab quaternion format [w, x, y, z]
        eef_quat_isaac = np.array([
            eef_quat_ros[1],  
            eef_quat_ros[2],  
            eef_quat_ros[3],  
            eef_quat_ros[0]   
        ])

        # Apply slight offset to the cube position
        offset = np.array([0.0, 0.0, -0.015])  # Slightly below the gripper
        attached_cube_pos = eef_pos + offset

        # Update the attached cube's position and orientation
        # E.g if cube_2 is attached -> self.cube_attached = 'cube_2'
        self.cube_positions[self.cube_attached] = attached_cube_pos
        self.cube_quaternions[self.cube_attached] = eef_quat_isaac


        
