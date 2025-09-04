import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path
import argparse

def load_cube_configurations(file_path):
    """Load cube configurations from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['configurations']

def plot_cube_configurations(configurations, save_path=None):
    """Plot all cube configurations in a grid layout."""
    
    # Calculate grid dimensions
    n_configs = len(configurations)
    cols = 4  # Use 4 columns for better space utilization
    rows = (n_configs + cols - 1) // cols
    
    # Create figure with better proportions
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
    fig.patch.set_facecolor('white')
    
    # Handle single row/column cases
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Enhanced colors and styling
    colors = ['#2E86C1', '#E74C3C', '#28B463']  # Blue, Red, Green
    cube_names = ['Cube 1', 'Cube 2', 'Cube 3']
    color_names = ['Blue', 'Red', 'Green']
    
    for idx, config in enumerate(configurations):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        # Extract positions
        poses = config['poses']
        x_positions = [pose['pos'][0] for pose in poses]
        y_positions = [pose['pos'][1] for pose in poses]
        
        # Plot workspace boundaries
        workspace_x = [0.35, 0.65]
        workspace_y = [-0.3, 0.3]
        
        # Draw workspace boundary
        workspace_rect = patches.Rectangle(
            (workspace_x[0], workspace_y[0]), 
            workspace_x[1] - workspace_x[0], 
            workspace_y[1] - workspace_y[0],
            linewidth=1.5, edgecolor='#7F8C8D', facecolor='#F8F9FA', alpha=0.3
        )
        ax.add_patch(workspace_rect)
        
        # Plot cubes as colored squares instead of circles
        cube_size = 0.05  # Size of the cube (width and height)
        for i, (x, y) in enumerate(zip(x_positions, y_positions)):
            # Create a square patch to represent a cube
            cube_rect = patches.Rectangle(
                (x - cube_size/2, y - cube_size/2),  # Bottom-left corner
                cube_size, cube_size,  # Width and height
                linewidth=1.5, 
                edgecolor='black', 
                facecolor=colors[i], 
                alpha=0.9,
                zorder=5
            )
            ax.add_patch(cube_rect)
        
        # Set equal aspect ratio and limits
        ax.set_xlim(workspace_x[0] - 0.05, workspace_x[1] + 0.05)
        ax.set_ylim(workspace_y[0] - 0.05, workspace_y[1] + 0.05)
        ax.set_aspect('equal')
        
        # Add custom grid with 0.05 spacing
        grid_spacing = 0.05
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        
        # Create grid lines
        x_grid = np.arange(np.floor(x_min/grid_spacing)*grid_spacing, 
                          np.ceil(x_max/grid_spacing)*grid_spacing + grid_spacing, 
                          grid_spacing)
        y_grid = np.arange(np.floor(y_min/grid_spacing)*grid_spacing, 
                          np.ceil(y_max/grid_spacing)*grid_spacing + grid_spacing, 
                          grid_spacing)
        
        # Draw vertical grid lines
        for x in x_grid:
            if x_min <= x <= x_max:
                ax.axvline(x, color='#BDC3C7', alpha=0.4, linewidth=0.5, zorder=1)
        
        # Draw horizontal grid lines
        for y in y_grid:
            if y_min <= y <= y_max:
                ax.axhline(y, color='#BDC3C7', alpha=0.4, linewidth=0.5, zorder=1)
        
        # Shorten the title: just show the config number (e.g., "#1")
        short_title = f"#{idx+1}"
        ax.set_title(short_title, fontsize=11, fontweight='bold', pad=8)

        # Minimal axis labels
        ax.set_xlabel('X (m)', fontsize=8)
        ax.set_ylabel('Y (m)', fontsize=8)
        
        # Remove the default grid since we're using custom grid
        # ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)  # Comment out this line
        ax.tick_params(labelsize=7)
        
        # Clean axes
        for spine in ax.spines.values():
            spine.set_linewidth(1)
            spine.set_color('#34495E')
    
    # Hide empty subplots
    for idx in range(n_configs, rows * cols):
        row = idx // cols
        col = idx % cols
        if row < rows and col < cols:
            axes[row, col].set_visible(False)
    
    # Clean main title
    plt.suptitle('Cube Configurations for Robotic Stacking Task', 
                fontsize=16, fontweight='bold', y=0.95)
    
    # Tight layout with proper spacing
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, hspace=0.5, wspace=0.3)
    
    if save_path:
        # Get file extension to determine primary format
        save_path = Path(save_path)
        suffix = save_path.suffix.lower()

        # Primary save
        if suffix == '.svg':
            plt.savefig(save_path, format='svg', bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            print(f"✅ Vector graphic (SVG) saved to {save_path}")
        elif suffix == '.pdf':
            plt.savefig(save_path, format='pdf', bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            print(f"✅ Vector graphic (PDF) saved to {save_path}")
        else:
            # Default raster (e.g., .png)
            plt.savefig(save_path, dpi=600, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            print(f"✅ Raster image saved to {save_path}")

        # Always also save a PDF (publication ready) unless the main file already is PDF
        if suffix != '.pdf':
            pdf_path = save_path.with_suffix('.pdf')
            try:
                plt.savefig(pdf_path, format='pdf', bbox_inches='tight',
                            facecolor='white', edgecolor='none')
                print(f"📄 Additional publication PDF saved to {pdf_path}")
            except Exception as e:
                print(f"⚠️ Could not save PDF version: {e}")

    plt.show()

def print_configuration_summary(configurations):
    """Print a summary of all configurations."""
    print("\n" + "="*60)
    print("📊 CUBE CONFIGURATION SUMMARY".center(60))
    print("="*60)
    
    for i, config in enumerate(configurations):
        print(f"\n{i+1:2d}. 🎯 {config['name']}")
        print(f"    📝 Description: {config['description']}")
        
        poses = config['poses']
        for j, pose in enumerate(poses):
            pos = pose['pos']
            print(f"    🧊 Cube {j+1}: x={pos[0]:6.3f}m, y={pos[1]:6.3f}m, z={pos[2]:6.4f}m")
    
    print("\n" + "="*60)

def analyze_workspace_coverage(configurations):
    """Analyze the workspace coverage of all configurations."""
    all_x = []
    all_y = []
    
    for config in configurations:
        for pose in config['poses']:
            all_x.append(pose['pos'][0])
            all_y.append(pose['pos'][1])
    
    print("\n" + "="*50)
    print("📏 WORKSPACE ANALYSIS".center(50))
    print("="*50)
    print(f"📐 X Range: {min(all_x):.3f}m to {max(all_x):.3f}m")
    print(f"📐 Y Range: {min(all_y):.3f}m to {max(all_y):.3f}m")
    print(f"📏 X Span:  {max(all_x) - min(all_x):.3f}m")
    print(f"📏 Y Span:  {max(all_y) - min(all_y):.3f}m")
    print(f"🔢 Total Configurations: {len(configurations)}")
    print(f"🧊 Total Cube Positions: {len(all_x)}")
    print("="*50)

def main():
    """Main function to run the visualization script."""
    parser = argparse.ArgumentParser(description="🎨 Visualize cube configurations from JSON file")
    parser.add_argument("--json", type=str, 
                       default="bc_stack_task_test_cases_extended.json",
                       help="Path to JSON configuration file")
    parser.add_argument("--output", type=str, 
                       default="cube_configurations_visualization.svg",
                       help="Output path for the visualization (.svg for vector, .png for raster)")
    
    args = parser.parse_args()
    json_file_path = args.json
    
    # Check if file exists
    if not Path(json_file_path).exists():
        print(f"❌ Error: File {json_file_path} not found!")
        print("Please make sure the path is correct.")
        return
    
    try:
        # Load configurations
        configurations = load_cube_configurations(json_file_path)
        print(f"✅ Loaded {len(configurations)} cube configurations")
        
        # Print summary
        print_configuration_summary(configurations)
        
        # Analyze workspace
        analyze_workspace_coverage(configurations)
        
        # Create and show plots
        plot_cube_configurations(configurations, save_path=args.output)
        
    except Exception as e:
        print(f"❌ Error loading or processing file: {e}")

if __name__ == "__main__":
    main()