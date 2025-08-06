#!/usr/bin/env python3
"""
EEF Trajectory and Quaternion Plotter for JSON Observation Data
Reads observation data from successful_stack_obs1.json and plots trajectories and quaternions using Plotly.
"""

import json
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.subplots import make_subplots
import numpy as np
import argparse
import os
import sys
from datetime import datetime

class JSONTrajectoryPlotter:
    def __init__(self, json_file=None):
        self.json_file = json_file
        self.data = None
        self.processed_data = {}
        
    def load_data(self):
        """Load JSON observation data"""
        if not os.path.exists(self.json_file):
            print(f"❌ JSON file not found: {self.json_file}")
            return False
        
        try:
            with open(self.json_file, 'r') as f:
                self.data = json.load(f)
            
            print(f"✅ Loaded JSON data with {len(self.data)} trials")
            
            # Process all trials
            for trial in self.data:
                trial_id = trial['trial']
                observations = trial['observations']
                
                if len(observations) == 0:
                    print(f"⚠️ Trial {trial_id} has no observations, skipping")
                    continue
                
                print(f"📊 Trial {trial_id}: {len(observations)} observations")
                
                # Extract EEF positions and quaternions
                eef_positions = []
                eef_quaternions = []
                timestamps = []
                
                for i, obs in enumerate(observations):
                    # Extract EEF position [x, y, z]
                    eef_pos = obs['eef_pos']
                    eef_positions.append(eef_pos)
                    
                    # Extract EEF quaternion [qw, qx, qy, qz] (IsaacLab format)
                    eef_quat = obs['eef_quat']
                    # Convert to standard [qx, qy, qz, qw] for plotting
                    eef_quat_standard = [eef_quat[1], eef_quat[2], eef_quat[3], eef_quat[0]]
                    eef_quaternions.append(eef_quat_standard)
                    
                    # Use step index and metadata frequency to calculate time
                    dt = trial['metadata']['dt']
                    timestamp = i * dt
                    timestamps.append(timestamp)
                
                # Store processed data
                self.processed_data[trial_id] = {
                    'eef_positions': np.array(eef_positions),
                    'eef_quaternions': np.array(eef_quaternions),
                    'timestamps': np.array(timestamps),
                    'metadata': trial['metadata']
                }
                
            return True
            
        except Exception as e:
            print(f"❌ Error loading JSON file: {e}")
            return False
    
    def calculate_trajectory_stats(self, trial_id):
        """Calculate trajectory statistics for a specific trial"""
        if trial_id not in self.processed_data:
            return {}
        
        trial_data = self.processed_data[trial_id]
        positions = trial_data['eef_positions']
        timestamps = trial_data['timestamps']
        
        stats = {
            'duration': timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0,
            'total_points': len(positions),
            'frequency': len(positions) / (timestamps[-1] - timestamps[0]) if len(timestamps) > 1 and timestamps[-1] > timestamps[0] else 0,
        }
        
        # Position ranges for each axis
        for i, axis in enumerate(['x', 'y', 'z']):
            axis_data = positions[:, i]
            stats[f'{axis}_range'] = [axis_data.min(), axis_data.max()]
            stats[f'{axis}_travel'] = axis_data.max() - axis_data.min()
        
        # Calculate total distance traveled
        if len(positions) > 1:
            distances = []
            for i in range(1, len(positions)):
                dist = np.linalg.norm(positions[i] - positions[i-1])
                distances.append(dist)
            
            stats['total_distance'] = sum(distances)
            stats['avg_speed'] = stats['total_distance'] / stats['duration'] if stats['duration'] > 0 else 0
        
        return stats
    
    def create_trajectory_plot(self, trial_id):
        """Create 3D trajectory plot for a specific trial"""
        if trial_id not in self.processed_data:
            return None
        
        trial_data = self.processed_data[trial_id]
        positions = trial_data['eef_positions']
        timestamps = trial_data['timestamps']
        
        fig = go.Figure()
        
        # Add 3D trajectory line
        fig.add_trace(go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode='lines+markers',
            line=dict(
                color=timestamps,
                colorscale='Viridis',
                width=4,
                colorbar=dict(title="Time (s)")
            ),
            marker=dict(
                size=3,
                color=timestamps,
                colorscale='Viridis',
                opacity=0.7
            ),
            name=f'EEF Trajectory (Trial {trial_id})',
            hovertemplate='<b>End-Effector Position</b><br>' +
                         'X: %{x:.4f} m<br>' +
                         'Y: %{y:.4f} m<br>' +
                         'Z: %{z:.4f} m<br>' +
                         'Time: %{marker.color:.2f} s<br>' +
                         '<extra></extra>'
        ))
        
        # Add start point
        fig.add_trace(go.Scatter3d(
            x=[positions[0, 0]],
            y=[positions[0, 1]],
            z=[positions[0, 2]],
            mode='markers',
            marker=dict(size=8, color='green', symbol='diamond'),
            name='Start',
            hovertemplate='<b>Start Position</b><br>' +
                         'X: %{x:.4f} m<br>' +
                         'Y: %{y:.4f} m<br>' +
                         'Z: %{z:.4f} m<br>' +
                         '<extra></extra>'
        ))
        
        # Add end point
        fig.add_trace(go.Scatter3d(
            x=[positions[-1, 0]],
            y=[positions[-1, 1]],
            z=[positions[-1, 2]],
            mode='markers',
            marker=dict(size=8, color='red', symbol='diamond'),
            name='End',
            hovertemplate='<b>End Position</b><br>' +
                         'X: %{x:.4f} m<br>' +
                         'Y: %{y:.4f} m<br>' +
                         'Z: %{z:.4f} m<br>' +
                         '<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title=f'End-Effector 3D Trajectory - Trial {trial_id}',
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Z (m)',
                aspectmode='data',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            height=600,
            showlegend=True
        )
        
        return fig
    
    def create_position_time_plot(self, trial_id):
        """Create position vs time plot for a specific trial"""
        if trial_id not in self.processed_data:
            return None
        
        trial_data = self.processed_data[trial_id]
        positions = trial_data['eef_positions']
        timestamps = trial_data['timestamps']
        
        fig = go.Figure()
        
        colors = ['red', 'green', 'blue']
        axes = ['x', 'y', 'z']
        
        for i, (axis, color) in enumerate(zip(axes, colors)):
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=positions[:, i],
                mode='lines',
                name=f'Position {axis.upper()}',
                line=dict(color=color, width=2),
                hovertemplate=f'<b>Position {axis.upper()}</b><br>' +
                             'Time: %{x:.2f} s<br>' +
                             f'{axis.upper()}: %{{y:.4f}} m<br>' +
                             '<extra></extra>'
            ))
        
        fig.update_layout(
            title=f'End-Effector Position vs Time - Trial {trial_id}',
            xaxis_title='Time (s)',
            yaxis_title='Position (m)',
            height=400,
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
    
    def create_quaternion_plot(self, trial_id):
        """Create quaternion components vs time plot for a specific trial"""
        if trial_id not in self.processed_data:
            return None
        
        trial_data = self.processed_data[trial_id]
        quaternions = trial_data['eef_quaternions']
        timestamps = trial_data['timestamps']
        
        fig = go.Figure()
        
        colors = ['red', 'green', 'blue', 'orange']
        components = ['x', 'y', 'z', 'w']
        
        for i, (comp, color) in enumerate(zip(components, colors)):
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=quaternions[:, i],
                mode='lines',
                name=f'q{comp}',
                line=dict(color=color, width=2),
                hovertemplate=f'<b>Quaternion q{comp}</b><br>' +
                             'Time: %{x:.2f} s<br>' +
                             f'q{comp}: %{{y:.4f}}<br>' +
                             '<extra></extra>'
            ))
        
        # Calculate and add quaternion magnitude
        magnitude = np.sqrt(np.sum(quaternions**2, axis=1))
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=magnitude,
            mode='lines',
            name='|q| (magnitude)',
            line=dict(color='black', width=2, dash='dash'),
            hovertemplate='<b>Quaternion Magnitude</b><br>' +
                         'Time: %{x:.2f} s<br>' +
                         '|q|: %{y:.4f}<br>' +
                         '<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'End-Effector Quaternion Components vs Time - Trial {trial_id}',
            xaxis_title='Time (s)',
            yaxis_title='Quaternion Component',
            height=400,
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
    
    def create_velocity_plot(self, trial_id):
        """Create velocity plot for a specific trial"""
        if trial_id not in self.processed_data:
            return None
        
        trial_data = self.processed_data[trial_id]
        positions = trial_data['eef_positions']
        timestamps = trial_data['timestamps']
        
        if len(positions) < 2:
            return None
        
        # Calculate velocities
        dt = np.diff(timestamps)
        velocities = np.diff(positions, axis=0) / dt[:, np.newaxis]
        time_mid = timestamps[:-1] + dt/2  # Midpoint times
        
        fig = go.Figure()
        
        colors = ['red', 'green', 'blue']
        axes = ['x', 'y', 'z']
        
        for i, (axis, color) in enumerate(zip(axes, colors)):
            fig.add_trace(go.Scatter(
                x=time_mid,
                y=velocities[:, i],
                mode='lines',
                name=f'Velocity {axis.upper()}',
                line=dict(color=color, width=2),
                hovertemplate=f'<b>Velocity {axis.upper()}</b><br>' +
                             'Time: %{x:.2f} s<br>' +
                             f'v_{axis}: %{{y:.4f}} m/s<br>' +
                             '<extra></extra>'
            ))
        
        # Calculate speed (magnitude of velocity)
        speed = np.sqrt(np.sum(velocities**2, axis=1))
        fig.add_trace(go.Scatter(
            x=time_mid,
            y=speed,
            mode='lines',
            name='Speed (|v|)',
            line=dict(color='black', width=3),
            hovertemplate='<b>Speed</b><br>' +
                         'Time: %{x:.2f} s<br>' +
                         '|v|: %{y:.4f} m/s<br>' +
                         '<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'End-Effector Velocity vs Time - Trial {trial_id}',
            xaxis_title='Time (s)',
            yaxis_title='Velocity (m/s)',
            height=400,
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
    
    def create_combined_dashboard(self, trial_id):
        """Create a combined dashboard with all plots for a specific trial"""
        if trial_id not in self.processed_data:
            return None
        
        trial_data = self.processed_data[trial_id]
        positions = trial_data['eef_positions']
        quaternions = trial_data['eef_quaternions']
        timestamps = trial_data['timestamps']
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Position vs Time', '3D Trajectory', 
                          'Quaternion vs Time', 'Velocity vs Time'),
            specs=[[{'type': 'scatter'}, {'type': 'scatter3d'}],
                   [{'type': 'scatter'}, {'type': 'scatter'}]],
            vertical_spacing=0.12,
            horizontal_spacing=0.08
        )
        
        # Add position vs time (row 1, col 1)
        colors = ['red', 'green', 'blue']
        for i, axis in enumerate(['x', 'y', 'z']):
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=positions[:, i],
                mode='lines',
                name=f'Pos {axis.upper()}',
                line=dict(color=colors[i], width=2),
                legendgroup='position'
            ), row=1, col=1)
        
        # Add 3D trajectory (row 1, col 2)
        fig.add_trace(go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode='lines+markers',
            line=dict(color=timestamps, colorscale='Viridis', width=3),
            marker=dict(size=2, color=timestamps, colorscale='Viridis'),
            name='Trajectory',
            showlegend=False
        ), row=1, col=2)
        
        # Add quaternion vs time (row 2, col 1)
        quat_colors = ['red', 'green', 'blue', 'orange']
        for i, comp in enumerate(['x', 'y', 'z', 'w']):
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=quaternions[:, i],
                mode='lines',
                name=f'q{comp}',
                line=dict(color=quat_colors[i], width=2),
                legendgroup='quaternion'
            ), row=2, col=1)
        
        # Add velocity vs time (row 2, col 2)
        if len(positions) > 1:
            dt = np.diff(timestamps)
            velocities = np.diff(positions, axis=0) / dt[:, np.newaxis]
            time_mid = timestamps[:-1] + dt/2
            
            for i, axis in enumerate(['x', 'y', 'z']):
                fig.add_trace(go.Scatter(
                    x=time_mid,
                    y=velocities[:, i],
                    mode='lines',
                    name=f'Vel {axis.upper()}',
                    line=dict(color=colors[i], width=2),
                    legendgroup='velocity'
                ), row=2, col=2)
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text=f"End-Effector Dynamics Dashboard - Trial {trial_id}",
            showlegend=True
        )
        
        # Update 3D scene
        fig.update_scenes(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data'
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Time (s)", row=1, col=1)
        fig.update_yaxes(title_text="Position (m)", row=1, col=1)
        
        fig.update_xaxes(title_text="Time (s)", row=2, col=1)
        fig.update_yaxes(title_text="Quaternion", row=2, col=1)
        
        fig.update_xaxes(title_text="Time (s)", row=2, col=2)
        fig.update_yaxes(title_text="Velocity (m/s)", row=2, col=2)
        
        return fig
    
    def create_multi_trial_comparison(self):
        """Create comparison plots showing all trials together"""
        if not self.processed_data:
            return None
        
        # Create subplots for comparison
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Position Comparison', '3D Trajectories', 
                          'Quaternion Comparison', 'Speed Comparison'),
            specs=[[{'type': 'scatter'}, {'type': 'scatter3d'}],
                   [{'type': 'scatter'}, {'type': 'scatter'}]],
            vertical_spacing=0.12,
            horizontal_spacing=0.08
        )
        
        # Color palette for different trials
        trial_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        
        for i, trial_id in enumerate(sorted(self.processed_data.keys())):
            trial_data = self.processed_data[trial_id]
            positions = trial_data['eef_positions']
            quaternions = trial_data['eef_quaternions']
            timestamps = trial_data['timestamps']
            color = trial_colors[i % len(trial_colors)]
            
            # Add Z position comparison (row 1, col 1)
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=positions[:, 2],  # Z coordinate
                mode='lines',
                name=f'Trial {trial_id} Z-pos',
                line=dict(color=color, width=2),
                legendgroup=f'trial_{trial_id}'
            ), row=1, col=1)
            
            # Add 3D trajectory (row 1, col 2)
            fig.add_trace(go.Scatter3d(
                x=positions[:, 0],
                y=positions[:, 1],
                z=positions[:, 2],
                mode='lines',
                name=f'Trial {trial_id}',
                line=dict(color=color, width=3),
                legendgroup=f'trial_{trial_id}',
                showlegend=False
            ), row=1, col=2)
            
            # Add quaternion W component (row 2, col 1)
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=quaternions[:, 3],  # W component
                mode='lines',
                name=f'Trial {trial_id} qw',
                line=dict(color=color, width=2),
                legendgroup=f'trial_{trial_id}',
                showlegend=False
            ), row=2, col=1)
            
            # Add speed comparison (row 2, col 2)
            if len(positions) > 1:
                dt = np.diff(timestamps)
                velocities = np.diff(positions, axis=0) / dt[:, np.newaxis]
                speed = np.sqrt(np.sum(velocities**2, axis=1))
                time_mid = timestamps[:-1] + dt/2
                
                fig.add_trace(go.Scatter(
                    x=time_mid,
                    y=speed,
                    mode='lines',
                    name=f'Trial {trial_id} speed',
                    line=dict(color=color, width=2),
                    legendgroup=f'trial_{trial_id}',
                    showlegend=False
                ), row=2, col=2)
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="Multi-Trial Comparison Dashboard",
            showlegend=True
        )
        
        # Update 3D scene
        fig.update_scenes(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data'
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Time (s)", row=1, col=1)
        fig.update_yaxes(title_text="Z Position (m)", row=1, col=1)
        
        fig.update_xaxes(title_text="Time (s)", row=2, col=1)
        fig.update_yaxes(title_text="Quaternion W", row=2, col=1)
        
        fig.update_xaxes(title_text="Time (s)", row=2, col=2)
        fig.update_yaxes(title_text="Speed (m/s)", row=2, col=2)
        
        return fig
    
    def print_statistics(self, trial_id=None):
        """Print trajectory statistics"""
        if trial_id is not None:
            # Print stats for specific trial
            if trial_id not in self.processed_data:
                print(f"❌ Trial {trial_id} not found")
                return
            
            stats = self.calculate_trajectory_stats(trial_id)
            trial_data = self.processed_data[trial_id]
            
            print("\n" + "="*70)
            print(f"TRAJECTORY STATISTICS - TRIAL {trial_id}")
            print("="*70)
            print(f"Task: {trial_data['metadata']['task']}")
            print(f"Duration: {stats['duration']:.2f} seconds")
            print(f"Total Points: {stats['total_points']}")
            print(f"Frequency: {stats['frequency']:.1f} Hz")
            
            if 'total_distance' in stats:
                print(f"Total Distance Traveled: {stats['total_distance']:.4f} m")
                print(f"Average Speed: {stats['avg_speed']:.4f} m/s")
            
            print(f"\nPosition Ranges:")
            for axis in ['x', 'y', 'z']:
                if f'{axis}_range' in stats:
                    range_vals = stats[f'{axis}_range']
                    travel = stats[f'{axis}_travel']
                    print(f"  {axis.upper()}: [{range_vals[0]:.4f}, {range_vals[1]:.4f}] m (travel: {travel:.4f} m)")
            
            # Quaternion statistics
            quaternions = trial_data['eef_quaternions']
            magnitude = np.sqrt(np.sum(quaternions**2, axis=1))
            print(f"\nQuaternion Statistics:")
            print(f"  Magnitude range: [{magnitude.min():.6f}, {magnitude.max():.6f}]")
            print(f"  Magnitude mean: {magnitude.mean():.6f}")
            print(f"  Magnitude std: {magnitude.std():.6f}")
            
            print("="*70)
        else:
            # Print summary for all trials
            print("\n" + "="*70)
            print("ALL TRIALS SUMMARY")
            print("="*70)
            for trial_id in sorted(self.processed_data.keys()):
                stats = self.calculate_trajectory_stats(trial_id)
                print(f"Trial {trial_id}: {stats['total_points']} points, {stats['duration']:.2f}s, {stats['frequency']:.1f} Hz")
            print("="*70)
    
    def plot_trial(self, trial_id, show_stats=True, save_html=None, show_plots=True):
        """Plot all visualizations for a specific trial"""
        if trial_id not in self.processed_data:
            print(f"❌ Trial {trial_id} not found")
            return False
        
        if show_stats:
            self.print_statistics(trial_id)
        
        print(f"\n📊 Creating plots for Trial {trial_id}...")
        
        # Create individual plots
        traj_fig = self.create_trajectory_plot(trial_id)
        pos_fig = self.create_position_time_plot(trial_id)
        quat_fig = self.create_quaternion_plot(trial_id)
        vel_fig = self.create_velocity_plot(trial_id)
        dashboard_fig = self.create_combined_dashboard(trial_id)
        
        # Save HTML files if requested
        if save_html:
            print(f"\n💾 Saving HTML files for Trial {trial_id}...")
            base_name = save_html.replace('.html', f'_trial_{trial_id}')
            
            if traj_fig:
                traj_fig.write_html(f"{base_name}_trajectory.html")
                print(f"  - {base_name}_trajectory.html")
            
            if pos_fig:
                pos_fig.write_html(f"{base_name}_position.html")
                print(f"  - {base_name}_position.html")
            
            if quat_fig:
                quat_fig.write_html(f"{base_name}_quaternion.html")
                print(f"  - {base_name}_quaternion.html")
            
            if vel_fig:
                vel_fig.write_html(f"{base_name}_velocity.html")
                print(f"  - {base_name}_velocity.html")
            
            if dashboard_fig:
                dashboard_fig.write_html(f"{base_name}_dashboard.html")
                print(f"  - {base_name}_dashboard.html")
        
        # Show plots
        if show_plots:
            print(f"\n🖥️ Displaying plots for Trial {trial_id}...")
            
            if dashboard_fig:
                print("  - Opening Combined Dashboard")
                dashboard_fig.show()
            
            if traj_fig:
                print("  - Opening 3D Trajectory")
                traj_fig.show()
        
        return True
    
    def plot_all_trials(self, show_stats=True, save_html=None, show_plots=True):
        """Plot visualizations for all trials and create comparison plots"""
        if not self.load_data():
            return False
        
        if show_stats:
            self.print_statistics()
        
        # Plot individual trials
        for trial_id in sorted(self.processed_data.keys()):
            self.plot_trial(trial_id, show_stats=False, save_html=save_html, show_plots=False)
        
        # Create multi-trial comparison
        print("\n📊 Creating multi-trial comparison...")
        comparison_fig = self.create_multi_trial_comparison()
        
        if save_html and comparison_fig:
            base_name = save_html.replace('.html', '')
            comparison_fig.write_html(f"{base_name}_comparison.html")
            print(f"💾 Saved: {base_name}_comparison.html")
        
        if show_plots and comparison_fig:
            print("🖥️ Opening Multi-Trial Comparison")
            comparison_fig.show()
        
        return True


def main():
    parser = argparse.ArgumentParser(description="Plot End-Effector Trajectory and Quaternion Data from JSON")
    parser.add_argument("--json", type=str, default="successful_stack_obs1.json",
                       help="Path to JSON observation file")
    parser.add_argument("--trial", type=int, default=None,
                       help="Specific trial ID to plot (if not provided, plots all trials)")
    parser.add_argument("--no-stats", action="store_true",
                       help="Don't print statistics")
    parser.add_argument("--save-html", type=str, default=None,
                       help="Save plots as HTML files (provide base filename)")
    parser.add_argument("--no-show", action="store_true",
                       help="Don't show interactive plots")
    
    args = parser.parse_args()
    
    try:
        print("🎯 JSON EEF Trajectory and Quaternion Plotter")
        print("="*50)
        
        # Create plotter
        plotter = JSONTrajectoryPlotter(json_file=args.json)
        
        if args.trial is not None:
            # Plot specific trial
            if not plotter.load_data():
                return 1
            
            success = plotter.plot_trial(
                trial_id=args.trial,
                show_stats=not args.no_stats,
                save_html=args.save_html,
                show_plots=not args.no_show
            )
        else:
            # Plot all trials
            success = plotter.plot_all_trials(
                show_stats=not args.no_stats,
                save_html=args.save_html,
                show_plots=not args.no_show
            )
        
        if success:
            print("\n✅ Plotting completed successfully!")
            if not args.no_show:
                print("💡 Close the browser tabs when done viewing the plots.")
        else:
            print("\n❌ Plotting failed!")
            return 1
            
    except KeyboardInterrupt:
        print("\n\n👋 Plotting interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())