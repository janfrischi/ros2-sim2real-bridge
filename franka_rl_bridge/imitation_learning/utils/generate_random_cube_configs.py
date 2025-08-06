import json
import numpy as np
import argparse
from pathlib import Path

def generate_random_cube_configurations(num_configs=16, workspace_x=(0.4, 0.6), workspace_y=(-0.25, 0.25), 
                                      z_height=0.0203, min_face_clearance=0.03, cube_size=0.05, 
                                      max_attempts=1000, grid_step=0.05):
    """
    Generate random cube configurations ensuring minimum face-to-face clearance between cubes.
    
    Args:
        num_configs: Number of configurations to generate
        workspace_x: Tuple of (min_x, max_x) for workspace bounds
        workspace_y: Tuple of (min_y, max_y) for workspace bounds
        z_height: Fixed z-height for all cubes
        min_face_clearance: Minimum clearance between cube faces (meters)
        cube_size: Size of each cube (width/length, assuming square) in meters
        max_attempts: Maximum attempts to find valid configuration
        grid_step: Discrete step size for x and y positions (in meters)
    
    Returns:
        Dictionary with configurations in the required format
    """
    # Calculate minimum center-to-center distance
    # For cubes not to touch: center_distance > cube_size
    # For minimum clearance: center_distance > cube_size + min_face_clearance
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
    
    configurations = []
    
    for config_idx in range(num_configs):
        attempts = 0
        valid_config = False

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
            print(f"⚠️  Warning: Could not find valid configuration {config_idx} after {max_attempts} attempts")
            print(f"   Consider: reducing cube size, increasing workspace, or reducing min clearance")
            # Still create the config but warn about potential issues
        
        # Create configuration in the required format
        config = {
            "name": f"config_{config_idx:02d}_random",
            "description": f"Random configuration {config_idx + 1}",
            "poses": []
        }
        
        for cube_idx, pos in enumerate(cube_positions):
            pose = {
                "pos": [round(pos[0], 3), round(pos[1], 3), round(pos[2], 4)],
                "quat": [0.0, 0.0, 0.0, 1.0]
            }
            config["poses"].append(pose)
        
        configurations.append(config)
    
    return {"configurations": configurations}

def print_config_summary(configurations_data, cube_size=0.05, min_face_clearance=0.03):
    """Print a summary of generated configurations with clearance analysis"""
    configs = configurations_data["configurations"]
    print(f"\n📊 Generated {len(configs)} random cube configurations")
    print("="*70)
    
    all_x = []
    all_y = []
    min_center_distances = []
    min_face_clearances = []
    
    for i, config in enumerate(configs):
        poses = config["poses"]
        x_positions = [pose["pos"][0] for pose in poses]
        y_positions = [pose["pos"][1] for pose in poses]
        
        all_x.extend(x_positions)
        all_y.extend(y_positions)
        
        # Calculate minimum center-to-center distance and face clearance
        positions = [[pose["pos"][0], pose["pos"][1]] for pose in poses]
        config_min_center_dist = float('inf')
        
        for j in range(len(positions)):
            for k in range(j + 1, len(positions)):
                pos1 = np.array(positions[j])
                pos2 = np.array(positions[k])
                center_dist = np.linalg.norm(pos1 - pos2)
                config_min_center_dist = min(config_min_center_dist, center_dist)
        
        # Calculate face-to-face clearance
        face_clearance = config_min_center_dist - cube_size
        min_center_distances.append(config_min_center_dist)
        min_face_clearances.append(face_clearance)
        
        # Status indicator
        status = "✅" if face_clearance >= min_face_clearance else "⚠️ "
        
        print(f"Config {i:2d}: Center dist = {config_min_center_dist:.3f}m, Face clearance = {face_clearance:.3f}m {status}")
        for j, pose in enumerate(poses):
            pos = pose["pos"]
            print(f"  🧊 Cube {j+1}: x={pos[0]:6.3f}m, y={pos[1]:6.3f}m, z={pos[2]:6.4f}m")
    
    print("\n" + "="*60)
    print("📏 WORKSPACE & CLEARANCE ANALYSIS")
    print("="*60)
    print(f"📐 X Range: {min(all_x):.3f}m to {max(all_x):.3f}m")
    print(f"📐 Y Range: {min(all_y):.3f}m to {max(all_y):.3f}m")
    print(f"📏 X Span:  {max(all_x) - min(all_x):.3f}m")
    print(f"📏 Y Span:  {max(all_y) - min(all_y):.3f}m")
    print(f"🔢 Total Configurations: {len(configs)}")
    print(f"🧊 Total Cube Positions: {len(all_x)}")
    
    # Clearance statistics
    print(f"\n🔍 CLEARANCE STATISTICS:")
    print(f"   Center Distance Range: {min(min_center_distances):.3f}m to {max(min_center_distances):.3f}m")
    print(f"   Face Clearance Range:  {min(min_face_clearances):.3f}m to {max(min_face_clearances):.3f}m")
    print(f"   Average Face Clearance: {np.mean(min_face_clearances):.3f}m")
    print(f"   Required Face Clearance: {min_face_clearance:.3f}m")
    
    # Count valid configurations
    valid_configs = sum(1 for clearance in min_face_clearances if clearance >= min_face_clearance)
    print(f"   Valid Configurations: {valid_configs}/{len(configs)} ({valid_configs/len(configs)*100:.1f}%)")
    
    print("="*60)

def main():
    """Main function to generate random cube configurations"""
    parser = argparse.ArgumentParser(description="🎲 Generate random cube configurations for robotic tasks")
    parser.add_argument("--output", type=str, default="random_cube_configurations.json",
                       help="Output JSON file path")
    parser.add_argument("--num-configs", type=int, default=16,
                       help="Number of configurations to generate")
    parser.add_argument("--workspace-x", nargs=2, type=float, default=[0.4, 0.6],
                       help="X workspace bounds (min max)")
    parser.add_argument("--workspace-y", nargs=2, type=float, default=[-0.25, 0.25],
                       help="Y workspace bounds (min max)")
    parser.add_argument("--min-face-clearance", type=float, default=0.03,
                       help="Minimum clearance between cube faces in meters")
    parser.add_argument("--cube-size", type=float, default=0.05,
                       help="Size of each cube (width/length) in meters")
    parser.add_argument("--z-height", type=float, default=0.0203,
                       help="Z height for all cubes")
    parser.add_argument("--grid-step", type=float, default=0.05,
                       help="Discrete step size for x and y positions (meters)")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducible results")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress summary output")
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"🎲 Using random seed: {args.seed}")
    
    print(f"🎯 Generating {args.num_configs} random cube configurations...")
    print(f"📐 Workspace: X=[{args.workspace_x[0]}, {args.workspace_x[1]}], Y=[{args.workspace_y[0]}, {args.workspace_y[1]}]")
    print(f"🧊 Cube size: {args.cube_size*100:.1f}cm x {args.cube_size*100:.1f}cm")
    print(f"📏 Minimum face clearance: {args.min_face_clearance*100:.1f}cm")
    print(f"🔲 Grid step size: {args.grid_step}m")
    
    # Generate configurations
    configurations_data = generate_random_cube_configurations(
        num_configs=args.num_configs,
        workspace_x=tuple(args.workspace_x),
        workspace_y=tuple(args.workspace_y),
        z_height=args.z_height,
        min_face_clearance=args.min_face_clearance,
        cube_size=args.cube_size,
        grid_step=args.grid_step
    )
    
    # Save to JSON file
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(configurations_data, f, indent=2)
    
    print(f"✅ Saved {len(configurations_data['configurations'])} configurations to {output_path}")
    
    # Print summary unless quiet mode
    if not args.quiet:
        print_config_summary(configurations_data, args.cube_size, args.min_face_clearance)
    
    print(f"\n🎉 Random cube configuration generation complete!")
    print(f"📁 Output file: {output_path.absolute()}")

if __name__ == "__main__":
    main()