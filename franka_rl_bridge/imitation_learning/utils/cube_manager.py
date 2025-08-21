import numpy as np

class CubeManagerMixin: 

    def randomly_spawn_cubes(self):
        """Randomly spawn the three cubes within the specified range using grid-based sampling"""
        print("\n🎲 RANDOMLY SPAWNING CUBES")
        print("=" * 50)
        
        # Configuration parameters (matching generate_random_cube_configs.py)
        workspace_x = (0.4, 0.6)
        workspace_y = (-0.25, 0.25)
        z_height = 0.0203
        min_face_clearance = 0.03
        cube_size = 0.05
        grid_step = 0.05
        max_attempts = 1000
        
        # Calculate minimum center-to-center distance
        min_center_distance = cube_size + min_face_clearance
        
        print(f"🔢 Cube constraints:")
        print(f"   Cube size: {cube_size*100:.1f}cm x {cube_size*100:.1f}cm")
        print(f"   Minimum face clearance: {min_face_clearance*100:.1f}cm")
        print(f"   Required center-to-center distance: {min_center_distance*100:.1f}cm")
        
        # Create discrete grids for x and y positions
        x_values = np.arange(workspace_x[0], workspace_x[1] + grid_step, grid_step)
        y_values = np.arange(workspace_y[0], workspace_y[1] + grid_step, grid_step)
        
        # Round to avoid floating point precision issues
        x_values = np.round(x_values, decimals=3)
        y_values = np.round(y_values, decimals=3)
        
        print(f"\n📐 Discrete grid created:")
        print(f"   X positions: {len(x_values)} steps from {x_values[0]:.3f} to {x_values[-1]:.3f}")
        print(f"   Y positions: {len(y_values)} steps from {y_values[0]:.3f} to {y_values[-1]:.3f}")
        print(f"   Total grid points: {len(x_values) * len(y_values)}")
        print(f"   Grid step size: {grid_step}m")
        
        # Attempt to find valid configuration
        attempts = 0
        valid_config = False
        cube_positions = []
        
        while not valid_config and attempts < max_attempts:
            attempts += 1
            
            # Sample 3 unique positions from the grid
            grid_points = [(x, y) for x in x_values for y in y_values]
            chosen_points = np.array(
                np.random.choice(len(grid_points), size=3, replace=False)
            )
            cube_positions = [
                [grid_points[i][0], grid_points[i][1], z_height] for i in chosen_points
            ]
            
            # Check minimum center-to-center distance constraint
            valid_config = True
            for i in range(len(cube_positions)):
                for j in range(i + 1, len(cube_positions)):
                    pos1 = np.array(cube_positions[i][:2])
                    pos2 = np.array(cube_positions[j][:2])
                    center_distance = np.linalg.norm(pos1 - pos2)
                    
                    # Check if cubes are too close (faces would touch or overlap)
                    if center_distance < min_center_distance:
                        valid_config = False
                        break
                if not valid_config:
                    break
    
        if not valid_config:
            print(f"⚠️ Warning: Could not find valid configuration after {max_attempts} attempts")
            print(f"   Using fallback positions with relaxed constraints")
            # Fallback: use safe positions if we can't find a valid random one
            cube_positions = [
                [0.45, -0.15, z_height],
                [0.5, 0.0, z_height],
                [0.55, 0.15, z_height]
            ]
        else:
            print(f"✅ Valid configuration found after {attempts} attempts")
        
        # Convert to the format expected by the cube manager
        cube_names = ['cube_1', 'cube_2', 'cube_3']
        new_positions = {}
        
        for i, cube_name in enumerate(cube_names):
            new_positions[cube_name] = np.array(cube_positions[i])
        
        # Update cube positions
        self.cube_positions.update(new_positions)
        
        # Reset orientations to identity
        for cube_name in cube_names:
            self.cube_quaternions[cube_name] = np.array([0.0, 0.0, 0.0, 1.0])
        
        # Display the changes with clearance analysis
        print("\n📍 NEW CUBE POSITIONS:")
        color_names = {
            'cube_1': 'Blue Cube  ',
            'cube_2': 'Red Cube   ',
            'cube_3': 'Green Cube '
        }
        
        for cube_name, pos in new_positions.items():
            print(f"  {color_names[cube_name]}: [{pos[0]:+7.4f}, {pos[1]:+7.4f}, {pos[2]:+7.4f}]")
        
        # Calculate and display clearance metrics
        distances = []
        face_clearances = []
        cube_positions_list = list(new_positions.values())
        
        for i in range(len(cube_positions_list)):
            for j in range(i+1, len(cube_positions_list)):
                center_dist = np.linalg.norm(cube_positions_list[i][:2] - cube_positions_list[j][:2])
                face_clearance = center_dist - cube_size
                distances.append(center_dist)
                face_clearances.append(face_clearance)
        
        min_center_distance_actual = min(distances)
        min_face_clearance_actual = min(face_clearances)
        
        print(f"\n📊 CLEARANCE ANALYSIS:")
        print(f"   Min center-to-center distance: {min_center_distance_actual:.4f}m")
        print(f"   Min face-to-face clearance: {min_face_clearance_actual:.4f}m")
        print(f"   Required face clearance: {min_face_clearance:.4f}m")
        
        # Status indicator
        status = "✅ VALID" if min_face_clearance_actual >= min_face_clearance else "⚠️ TIGHT"
        print(f"   Configuration status: {status}")
        
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
                'cube_1': np.array([0.40, 0.0, 0.0203]),    # Input
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
            # Random configuration 1
            'cube_1': np.array([0.6, -0.0, 0.0203]),
            'cube_2': np.array([0.55, 0.1, 0.0203]),
            'cube_3': np.array([0.45, -0.25, 0.0203]),
        },
        "config_1": {
            # Random configuration 2
            'cube_1': np.array([0.4, -0.0, 0.0203]),
            'cube_2': np.array([0.6, -0.05, 0.0203]),
            'cube_3': np.array([0.4, -0.15, 0.0203]),
        },
        "config_2": {
            # Random configuration 3
            'cube_1': np.array([0.4, 0.1, 0.0203]),
            'cube_2': np.array([0.45, -0.15, 0.0203]),
            'cube_3': np.array([0.55, 0.05, 0.0203]),
        },
        "config_3": {
            # Random configuration 4
            'cube_1': np.array([0.55, 0.15, 0.0203]),
            'cube_2': np.array([0.5, -0.15, 0.0203]),
            'cube_3': np.array([0.5, -0.0, 0.0203]),
        },
        "config_4": {
            # Random configuration 5
            'cube_1': np.array([0.45, -0.25, 0.0203]),
            'cube_2': np.array([0.55, -0.15, 0.0203]),
            'cube_3': np.array([0.4, -0.05, 0.0203]),
        },
        "config_5": {
            # Random configuration 6
            'cube_1': np.array([0.55, 0.1, 0.0203]),
            'cube_2': np.array([0.5, -0.0, 0.0203]),
            'cube_3': np.array([0.4, -0.2, 0.0203]),
        },
        "config_6": {
            # Random configuration 7
            'cube_1': np.array([0.6, -0.2, 0.0203]),
            'cube_2': np.array([0.5, 0.0, 0.0203]),
            'cube_3': np.array([0.6, 0.1, 0.0203]),
        },
        "config_7": {
            # Random configuration 8
            'cube_1': np.array([0.55, 0.25, 0.0203]),
            'cube_2': np.array([0.45, -0.25, 0.0203]),
            'cube_3': np.array([0.6, 0.05, 0.0203]),
        },
        "config_8": {
            # Random configuration 9
            'cube_1': np.array([0.5, -0.1, 0.0203]),
            'cube_2': np.array([0.4, -0.05, 0.0203]),
            'cube_3': np.array([0.55, 0.15, 0.0203]),
        },
        "config_9": {
            # Random configuration 10
            'cube_1': np.array([0.6, -0.25, 0.0203]),
            'cube_2': np.array([0.55, 0.2, 0.0203]),
            'cube_3': np.array([0.5, 0.1, 0.0203]),
        },
        "config_10": {
            # Random configuration 11
            'cube_1': np.array([0.55, -0.1, 0.0203]),
            'cube_2': np.array([0.45, 0.05, 0.0203]),
            'cube_3': np.array([0.6, 0.2, 0.0203]),
        },
        "config_11": {
            # Random configuration 12
            'cube_1': np.array([0.6, -0.1, 0.0203]),
            'cube_2': np.array([0.45, -0.2, 0.0203]),
            'cube_3': np.array([0.45, 0.1, 0.0203]),
        },
        "config_12": {
            # Random configuration 13
            'cube_1': np.array([0.5, 0.25, 0.0203]),
            'cube_2': np.array([0.45, -0.15, 0.0203]),
            'cube_3': np.array([0.45, 0.05, 0.0203]),
        },
        "config_13": {
            # Random configuration 14
            'cube_1': np.array([0.45, -0.05, 0.0203]),
            'cube_2': np.array([0.6, 0.2, 0.0203]),
            'cube_3': np.array([0.55, -0.25, 0.0203]),
        },
        "config_14": {
            # Random configuration 15
            'cube_1': np.array([0.5, -0.25, 0.0203]),
            'cube_2': np.array([0.5, 0.15, 0.0203]),
            'cube_3': np.array([0.6, -0.25, 0.0203]),
        },
        "config_15": {
            # Random configuration 16
            'cube_1': np.array([0.4, -0.25, 0.0203]),
            'cube_2': np.array([0.6, -0.1, 0.0203]),
            'cube_3': np.array([0.6, 0.2, 0.0203]),
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

        attached_cube_pos = eef_pos

        # Update the attached cube's position and orientation
        # E.g if cube_2 is attached -> self.cube_attached = 'cube_2'
        self.cube_positions[self.cube_attached] = attached_cube_pos
        self.cube_quaternions[self.cube_attached] = eef_quat_isaac

    def update_cubes_from_camera(self):
        """Update cube positions using camera-detected poses - ONE-TIME INITIAL DETECTION"""
        print("\n📷 UPDATING CUBES FROM CAMERA (Initial Detection)")
        print("=" * 50)
        
        if not hasattr(self, 'camera_poses_received') or not self.camera_poses_received:
            print("❌ No camera poses received yet!")
            print("   Please ensure the camera detection system is running")
            print("   and publishing to /cube_poses/panda_link_cube_X topics")
            if hasattr(self, 'received_cubes'):
                print(f"   Received cubes: {list(self.received_cubes)}")
                print(f"   Expected cubes: {self.expected_cubes}")
            return False
        
        if not hasattr(self, 'camera_cube_poses') or not self.camera_cube_poses:
            print("❌ Camera pose data is empty!")
            return False
        
        # Check if camera positions have already been applied
        if hasattr(self, 'camera_positions_applied') and self.camera_positions_applied:
            print("📋 Camera positions have already been applied to cubes")
            print("   This is a ONE-TIME initial detection system")
            print("   💡 Use 'C' for new random positions or number keys for presets")
            print("   💡 Dynamic tracking is handled automatically by update_attached_cube_pose()")
            return False
        
        # Count how many cubes were detected
        detected_count = len(self.camera_cube_poses)
        print(f"📊 Camera detected {detected_count} cubes")
        
        # Set INITIAL cube positions based on camera detection
        updated_cubes = []
        
        # Colored emojis for each cube
        cube_colors = {
            'cube_1': '🔵',  # Blue
            'cube_2': '🔴',  # Red
            'cube_3': '🟢'   # Green
        }
        
        for cube_name in ['cube_1', 'cube_2', 'cube_3']:
            if cube_name in self.camera_cube_poses:
                pose_data = self.camera_cube_poses[cube_name]
                
                # Set INITIAL position and quaternion (one-time only)
                self.cube_positions[cube_name] = pose_data['position']
                self.cube_quaternions[cube_name] = pose_data['quaternion']
                
                # Display current position with colored emoji
                pos = pose_data['position']
                print(f"{cube_colors[cube_name]} {cube_name}: [{pos[0]:+7.4f}, {pos[1]:+7.4f}, {pos[2]:+7.4f}]")
                updated_cubes.append(cube_name)
            else:
                print(f"⚠️ No camera data for {cube_name}")
        
        if updated_cubes:
            # Mark that camera positions have been applied (prevents re-application)
            self.camera_positions_applied = True
            self.current_config_name = "Camera Initial Detection"
            
            print(f"\n✅ Set initial positions for {len(updated_cubes)} cube(s) from camera data")
            print("🔒 Camera detection complete - future cube tracking handled dynamically")
            print("=" * 50)
            
            # Reset policy episode state since environment changed
            if hasattr(self, 'policy') and self.policy:
                self.policy.start_episode()
                print("🔄 Policy episode state reset due to environment change")
            
            return True
        else:
            print("❌ No cubes were updated!")
            return False

    def reset_camera_detection(self, clear_poses=True, reset_flags=True):
        """Reset camera detection with options for granular control"""
        print("\n🔄 RESETTING CAMERA DETECTION STATE")
        print("=" * 50)
        
        if reset_flags:
            self.camera_positions_applied = False
            self.camera_poses_received = False
            print("📋 Reset application and reception flags")
        
        if clear_poses and hasattr(self, 'camera_cube_poses'):
            count = len(self.camera_cube_poses)
            self.camera_cube_poses.clear()
            print(f"🗑️ Cleared {count} stored camera poses")
        
        if hasattr(self, 'received_cubes'):
            self.received_cubes.clear()
            print("📦 Cleared received cubes tracking")
        
        print("✅ Camera detection reset complete!")
        return True