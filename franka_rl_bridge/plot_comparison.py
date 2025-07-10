#!/usr/bin/env python3
"""
Simple EEF Trajectory and Action Comparison Script
Compares end-effector position, quaternion, and action data from CSV (real robot) and JSON (simulation) files.

Usage:
    python3 plot_comparison.py --csv_file path/to/robot_data.csv --json_file path/to/simulation_data.json --trial 0

"""

import pandas as pd
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import argparse
import os
import sys

class EEFDataComparison:
    def __init__(self, csv_file, json_file, trial_id=0):
        self.csv_file = csv_file
        self.json_file = json_file
        self.trial_id = trial_id
        self.csv_data = None
        self.json_data = None
        
    def load_csv_data(self):
        """Load CSV data from real robot"""
        try:
            self.csv_data = pd.read_csv(self.csv_file)
            
            # Convert timestamp to relative time
            if 'timestamp' in self.csv_data.columns:
                self.csv_data['timestamp'] = pd.to_datetime(self.csv_data['timestamp'])
                start_time = self.csv_data['timestamp'].iloc[0]
                self.csv_data['time_seconds'] = (self.csv_data['timestamp'] - start_time).dt.total_seconds()
            else:
                # Use step count as time proxy (assuming 20Hz)
                self.csv_data['time_seconds'] = self.csv_data['step_count'] * 0.05
                
            print(f"✅ Loaded CSV data: {len(self.csv_data)} points")
            
            # Check if action data is available
            action_cols = [col for col in self.csv_data.columns if col.startswith('action_')]
            if action_cols:
                print(f"   Found action columns: {action_cols}")
            else:
                print("   ⚠️ No action columns found in CSV")
                
            return True
            
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            return False
    
    def load_json_data(self):
        """Load JSON data from simulation"""
        try:
            with open(self.json_file, 'r') as f:
                data = json.load(f)
            
            # Find the specified trial
            trial_data = None
            for trial in data:
                if trial['trial'] == self.trial_id:
                    trial_data = trial
                    break
            
            if trial_data is None:
                print(f"❌ Trial {self.trial_id} not found in JSON")
                return False
            
            observations = trial_data['observations']
            if len(observations) == 0:
                print(f"❌ Trial {self.trial_id} has no observations")
                return False
            
            # Extract EEF data and actions
            eef_positions = []
            eef_quaternions = []
            actions = []
            timestamps = []
            
            dt = trial_data['metadata']['dt']
            
            for i, obs in enumerate(observations):
                # Extract EEF position [x, y, z]
                eef_pos = obs['eef_pos']
                eef_positions.append(eef_pos)
                
                # Extract EEF quaternion [qw, qx, qy, qz] (IsaacLab format)
                # Convert to [qx, qy, qz, qw] to match CSV format
                eef_quat = obs['eef_quat']
                eef_quat_converted = [eef_quat[1], eef_quat[2], eef_quat[3], eef_quat[0]]
                eef_quaternions.append(eef_quat_converted)
                
                # Extract action if available
                if 'action' in obs:
                    action = obs['action']
                    actions.append(action)
                else:
                    # Create dummy action if not available
                    actions.append([0.0] * 8)
                
                # Calculate time
                timestamps.append(i * dt)
            
            # Create DataFrame for easier handling
            data_dict = {
                'time_seconds': timestamps,
                'eef_pos_x': [pos[0] for pos in eef_positions],
                'eef_pos_y': [pos[1] for pos in eef_positions],
                'eef_pos_z': [pos[2] for pos in eef_positions],
                'eef_quat_x': [quat[0] for quat in eef_quaternions],
                'eef_quat_y': [quat[1] for quat in eef_quaternions],
                'eef_quat_z': [quat[2] for quat in eef_quaternions],
                'eef_quat_w': [quat[3] for quat in eef_quaternions]
            }
            
            # Add action data if available
            if len(actions) > 0 and len(actions[0]) >= 8:
                data_dict.update({
                    'action_x': [action[0] for action in actions],
                    'action_y': [action[1] for action in actions],
                    'action_z': [action[2] for action in actions],
                    'action_quat_x': [action[4] for action in actions],  # Note: action format is [x,y,z,qw,qx,qy,qz,gripper]
                    'action_quat_y': [action[5] for action in actions],
                    'action_quat_z': [action[6] for action in actions],
                    'action_quat_w': [action[3] for action in actions],
                    'action_gripper': [action[7] for action in actions]
                })
                print(f"   Found action data with {len(actions[0])} components")
            else:
                print("   ⚠️ No action data found in JSON")
            
            self.json_data = pd.DataFrame(data_dict)
            
            print(f"✅ Loaded JSON data (Trial {self.trial_id}): {len(self.json_data)} points")
            return True
            
        except Exception as e:
            print(f"❌ Error loading JSON: {e}")
            return False
    
    def create_comparison_plot(self):
        """Create side-by-side comparison plots including actions"""
        if self.csv_data is None or self.json_data is None:
            print("❌ Data not loaded")
            return None
        
        # Check if action data is available in both datasets
        csv_has_actions = any(col.startswith('action_') for col in self.csv_data.columns)
        json_has_actions = any(col.startswith('action_') for col in self.json_data.columns)
        
        # Determine number of subplots based on available data
        if csv_has_actions and json_has_actions:
            n_rows = 3
            subplot_titles = (
                'End-Effector Position Comparison',
                'End-Effector Quaternion Comparison',
                'Action Comparison'
            )
        else:
            n_rows = 2
            subplot_titles = (
                'End-Effector Position Comparison',
                'End-Effector Quaternion Comparison'
            )
            if not csv_has_actions or not json_has_actions:
                print("⚠️ Action data not available in both datasets - skipping action comparison")
        
        # Create subplots
        fig = make_subplots(
            rows=n_rows, cols=1,
            subplot_titles=subplot_titles,
            vertical_spacing=0.1
        )
        
        # Colors for axes
        pos_colors = ['red', 'green', 'blue']
        quat_colors = ['red', 'green', 'blue', 'orange']
        action_colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        # Plot Position Data
        axes = ['x', 'y', 'z']
        for i, (axis, color) in enumerate(zip(axes, pos_colors)):
            # CSV data (real robot)
            fig.add_trace(go.Scatter(
                x=self.csv_data['time_seconds'],
                y=self.csv_data[f'eef_pos_{axis}'],
                mode='lines',
                name=f'Real Robot - Pos {axis.upper()}',
                line=dict(color=color, width=2),
                legendgroup='real_pos'
            ), row=1, col=1)
            
            # JSON data (simulation)
            fig.add_trace(go.Scatter(
                x=self.json_data['time_seconds'],
                y=self.json_data[f'eef_pos_{axis}'],
                mode='lines',
                name=f'Simulation - Pos {axis.upper()}',
                line=dict(color=color, width=2, dash='dash'),
                legendgroup='sim_pos'
            ), row=1, col=1)
        
        # Plot Quaternion Data
        quat_components = ['x', 'y', 'z', 'w']
        for i, (comp, color) in enumerate(zip(quat_components, quat_colors)):
            # CSV data (real robot)
            fig.add_trace(go.Scatter(
                x=self.csv_data['time_seconds'],
                y=self.csv_data[f'eef_quat_{comp}'],
                mode='lines',
                name=f'Real Robot - q{comp}',
                line=dict(color=color, width=2),
                legendgroup='real_quat'
            ), row=2, col=1)
            
            # JSON data (simulation)
            fig.add_trace(go.Scatter(
                x=self.json_data['time_seconds'],
                y=self.json_data[f'eef_quat_{comp}'],
                mode='lines',
                name=f'Simulation - q{comp}',
                line=dict(color=color, width=2, dash='dash'),
                legendgroup='sim_quat'
            ), row=2, col=1)
        
        # Plot Action Data (if available)
        if csv_has_actions and json_has_actions and n_rows == 3:
            # Action position components
            for i, (axis, color) in enumerate(zip(axes, pos_colors)):
                if f'action_{axis}' in self.csv_data.columns and f'action_{axis}' in self.json_data.columns:
                    # CSV data (real robot)
                    fig.add_trace(go.Scatter(
                        x=self.csv_data['time_seconds'],
                        y=self.csv_data[f'action_{axis}'],
                        mode='lines',
                        name=f'Real Robot - Action Pos {axis.upper()}',
                        line=dict(color=color, width=2),
                        legendgroup='real_action'
                    ), row=3, col=1)
                    
                    # JSON data (simulation)
                    fig.add_trace(go.Scatter(
                        x=self.json_data['time_seconds'],
                        y=self.json_data[f'action_{axis}'],
                        mode='lines',
                        name=f'Simulation - Action Pos {axis.upper()}',
                        line=dict(color=color, width=2, dash='dash'),
                        legendgroup='sim_action'
                    ), row=3, col=1)
            
            # Action quaternion components
            for i, (comp, color) in enumerate(zip(quat_components, quat_colors)):
                if f'action_quat_{comp}' in self.csv_data.columns and f'action_quat_{comp}' in self.json_data.columns:
                    # CSV data (real robot)
                    fig.add_trace(go.Scatter(
                        x=self.csv_data['time_seconds'],
                        y=self.csv_data[f'action_quat_{comp}'],
                        mode='lines',
                        name=f'Real Robot - Action q{comp}',
                        line=dict(color=quat_colors[i], width=1, dash='dot'),
                        legendgroup='real_action',
                        opacity=0.7
                    ), row=3, col=1)
                    
                    # JSON data (simulation)
                    fig.add_trace(go.Scatter(
                        x=self.json_data['time_seconds'],
                        y=self.json_data[f'action_quat_{comp}'],
                        mode='lines',
                        name=f'Simulation - Action q{comp}',
                        line=dict(color=quat_colors[i], width=1, dash='dashdot'),
                        legendgroup='sim_action',
                        opacity=0.7
                    ), row=3, col=1)
            
            # Action gripper component
            if 'action_gripper' in self.csv_data.columns and 'action_gripper' in self.json_data.columns:
                # CSV data (real robot)
                fig.add_trace(go.Scatter(
                    x=self.csv_data['time_seconds'],
                    y=self.csv_data['action_gripper'],
                    mode='lines',
                    name='Real Robot - Action Gripper',
                    line=dict(color='black', width=3),
                    legendgroup='real_action'
                ), row=3, col=1)
                
                # JSON data (simulation)
                fig.add_trace(go.Scatter(
                    x=self.json_data['time_seconds'],
                    y=self.json_data['action_gripper'],
                    mode='lines',
                    name='Simulation - Action Gripper',
                    line=dict(color='black', width=3, dash='dash'),
                    legendgroup='sim_action'
                ), row=3, col=1)
        
        # Update layout
        fig.update_layout(
            height=800 if n_rows == 2 else 1200,
            title_text=f"EEF Trajectory and Action Comparison: Real Robot vs Simulation (Trial {self.trial_id})",
            showlegend=True,
            hovermode='x unified'
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Time (s)", row=1, col=1)
        fig.update_yaxes(title_text="Position (m)", row=1, col=1)
        
        fig.update_xaxes(title_text="Time (s)", row=2, col=1)
        fig.update_yaxes(title_text="Quaternion Component", row=2, col=1)
        
        if n_rows == 3:
            fig.update_xaxes(title_text="Time (s)", row=3, col=1)
            fig.update_yaxes(title_text="Action Value", row=3, col=1)
        
        return fig
    
    def print_comparison_stats(self):
        """Print basic statistics comparison including actions"""
        if self.csv_data is None or self.json_data is None:
            return
        
        print("\n" + "="*80)
        print("COMPARISON STATISTICS")
        print("="*80)
        
        print(f"Real Robot Data:")
        print(f"  Duration: {self.csv_data['time_seconds'].iloc[-1]:.2f} seconds")
        print(f"  Points: {len(self.csv_data)}")
        print(f"  Frequency: {len(self.csv_data) / self.csv_data['time_seconds'].iloc[-1]:.1f} Hz")
        
        print(f"\nSimulation Data (Trial {self.trial_id}):")
        print(f"  Duration: {self.json_data['time_seconds'].iloc[-1]:.2f} seconds")
        print(f"  Points: {len(self.json_data)}")
        print(f"  Frequency: {len(self.json_data) / self.json_data['time_seconds'].iloc[-1]:.1f} Hz")
        
        # Position ranges
        print(f"\nPosition Ranges:")
        for axis in ['x', 'y', 'z']:
            csv_col = f'eef_pos_{axis}'
            csv_range = [self.csv_data[csv_col].min(), self.csv_data[csv_col].max()]
            json_range = [self.json_data[csv_col].min(), self.json_data[csv_col].max()]
            
            print(f"  {axis.upper()}: Real [{csv_range[0]:.4f}, {csv_range[1]:.4f}] | "
                  f"Sim [{json_range[0]:.4f}, {json_range[1]:.4f}]")
        
        # Quaternion magnitude check
        csv_quat_mag = np.sqrt(
            self.csv_data['eef_quat_x']**2 + self.csv_data['eef_quat_y']**2 + 
            self.csv_data['eef_quat_z']**2 + self.csv_data['eef_quat_w']**2
        )
        json_quat_mag = np.sqrt(
            self.json_data['eef_quat_x']**2 + self.json_data['eef_quat_y']**2 + 
            self.json_data['eef_quat_z']**2 + self.json_data['eef_quat_w']**2
        )
        
        print(f"\nQuaternion Magnitude:")
        print(f"  Real Robot: mean={csv_quat_mag.mean():.6f}, std={csv_quat_mag.std():.6f}")
        print(f"  Simulation: mean={json_quat_mag.mean():.6f}, std={json_quat_mag.std():.6f}")
        
        # Action statistics (if available)
        csv_has_actions = any(col.startswith('action_') for col in self.csv_data.columns)
        json_has_actions = any(col.startswith('action_') for col in self.json_data.columns)
        
        if csv_has_actions and json_has_actions:
            print(f"\nAction Statistics:")
            
            # Action position ranges
            for axis in ['x', 'y', 'z']:
                action_col = f'action_{axis}'
                if action_col in self.csv_data.columns and action_col in self.json_data.columns:
                    csv_range = [self.csv_data[action_col].min(), self.csv_data[action_col].max()]
                    json_range = [self.json_data[action_col].min(), self.json_data[action_col].max()]
                    
                    print(f"  Action {axis.upper()}: Real [{csv_range[0]:.4f}, {csv_range[1]:.4f}] | "
                          f"Sim [{json_range[0]:.4f}, {json_range[1]:.4f}]")
            
            # Action gripper range
            if 'action_gripper' in self.csv_data.columns and 'action_gripper' in self.json_data.columns:
                csv_gripper_range = [self.csv_data['action_gripper'].min(), self.csv_data['action_gripper'].max()]
                json_gripper_range = [self.json_data['action_gripper'].min(), self.json_data['action_gripper'].max()]
                
                print(f"  Action Gripper: Real [{csv_gripper_range[0]:.4f}, {csv_gripper_range[1]:.4f}] | "
                      f"Sim [{json_gripper_range[0]:.4f}, {json_gripper_range[1]:.4f}]")
        elif csv_has_actions:
            print(f"\n⚠️ Action data only available in CSV (Real Robot)")
        elif json_has_actions:
            print(f"\n⚠️ Action data only available in JSON (Simulation)")
        else:
            print(f"\n⚠️ No action data available in either dataset")
        
        print("="*80)
    
    def run_comparison(self, show_stats=True, save_html=None, show_plot=True):
        """Run the complete comparison"""
        print("🔄 Loading data...")
        
        if not self.load_csv_data():
            return False
        
        if not self.load_json_data():
            return False
        
        if show_stats:
            self.print_comparison_stats()
        
        print("\n📊 Creating comparison plot...")
        fig = self.create_comparison_plot()
        
        if fig is None:
            return False
        
        if save_html:
            fig.write_html(save_html)
            print(f"💾 Saved plot as: {save_html}")
        
        if show_plot:
            print("🖥️ Opening interactive plot...")
            fig.show()
        
        return True

def main():
    parser = argparse.ArgumentParser(description="Compare EEF trajectories and actions from CSV and JSON data")
    parser.add_argument("--csv", type=str, default="eef_dynamics_20250702_165452.csv",
                       help="Path to CSV file (real robot data)")
    parser.add_argument("--json", type=str, default="successful_stack_obs1.json",
                       help="Path to JSON file (simulation data)")
    parser.add_argument("--trial", type=int, default=0,
                       help="Trial ID to use from JSON file")
    parser.add_argument("--no-stats", action="store_true",
                       help="Don't print comparison statistics")
    parser.add_argument("--save-html", type=str, default=None,
                       help="Save plot as HTML file")
    parser.add_argument("--no-show", action="store_true",
                       help="Don't show interactive plot")
    
    args = parser.parse_args()
    
    try:
        print("🎯 EEF Trajectory and Action Comparison Tool")
        print("="*50)
        
        # Check if files exist
        if not os.path.exists(args.csv):
            print(f"❌ CSV file not found: {args.csv}")
            return 1
        
        if not os.path.exists(args.json):
            print(f"❌ JSON file not found: {args.json}")
            return 1
        
        # Create comparison object
        comparator = EEFDataComparison(args.csv, args.json, args.trial)
        
        # Run comparison
        success = comparator.run_comparison(
            show_stats=not args.no_stats,
            save_html=args.save_html,
            show_plot=not args.no_show
        )
        
        if success:
            print("\n✅ Comparison completed successfully!")
            if not args.no_show:
                print("💡 Close the browser tab when done viewing the plot.")
        else:
            print("\n❌ Comparison failed!")
            return 1
            
    except KeyboardInterrupt:
        print("\n\n👋 Comparison interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())