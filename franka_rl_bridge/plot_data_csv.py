#!/usr/bin/env python3
"""
EEF Trajectory and Quaternion Plotter
Reads CSV data from BC policy runner and plots trajectories and quaternions using Plotly.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.subplots import make_subplots
import numpy as np
import argparse
import os
import glob
from datetime import datetime
import sys

class EEFTrajectoryPlotter:
    def __init__(self, csv_file=None):
        self.csv_file = csv_file
        self.data = None
        
    def find_latest_csv(self, data_dir=None):
        """Find the latest EEF dynamics CSV file"""
        if data_dir is None:
            data_dir = os.path.join(os.path.expanduser("~"), "bc_policy_data")
        
        if not os.path.exists(data_dir):
            print(f"❌ Data directory not found: {data_dir}")
            return None
        
        # Find all EEF dynamics CSV files
        pattern = os.path.join(data_dir, "eef_dynamics_*.csv")
        csv_files = glob.glob(pattern)
        
        if not csv_files:
            print(f"❌ No EEF dynamics CSV files found in: {data_dir}")
            return None
        
        # Sort by modification time and get the latest
        latest_file = max(csv_files, key=os.path.getmtime)
        print(f"✅ Found latest CSV file: {os.path.basename(latest_file)}")
        return latest_file
    
    def load_data(self):
        """Load CSV data"""
        if self.csv_file is None:
            self.csv_file = self.find_latest_csv()
            if self.csv_file is None:
                return False
        
        if not os.path.exists(self.csv_file):
            print(f"❌ CSV file not found: {self.csv_file}")
            return False
        
        try:
            self.data = pd.read_csv(self.csv_file)
            print(f"✅ Loaded {len(self.data)} data points from CSV")
            
            # Convert timestamp to datetime if it's a string
            if 'timestamp' in self.data.columns:
                try:
                    self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
                except:
                    print("⚠️ Could not parse timestamp column")
            
            # Calculate time relative to start
            if len(self.data) > 0:
                if 'timestamp' in self.data.columns:
                    start_time = self.data['timestamp'].iloc[0]
                    self.data['time_seconds'] = (self.data['timestamp'] - start_time).dt.total_seconds()
                else:
                    # Use step count as time proxy
                    self.data['time_seconds'] = self.data['step_count'] * 0.05  # Assuming 20Hz
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading CSV file: {e}")
            return False
    
    def calculate_trajectory_stats(self):
        """Calculate trajectory statistics"""
        if self.data is None or len(self.data) == 0:
            return {}
        
        # Position statistics
        pos_stats = {
            'duration': self.data['time_seconds'].iloc[-1] - self.data['time_seconds'].iloc[0],
            'total_points': len(self.data),
            'frequency': len(self.data) / (self.data['time_seconds'].iloc[-1] - self.data['time_seconds'].iloc[0]) if len(self.data) > 1 else 0,
        }
        
        # Position ranges
        for axis in ['x', 'y', 'z']:
            col = f'eef_pos_{axis}'
            if col in self.data.columns:
                pos_stats[f'{axis}_range'] = [self.data[col].min(), self.data[col].max()]
                pos_stats[f'{axis}_travel'] = self.data[col].max() - self.data[col].min()
        
        # Calculate total distance traveled
        if all(f'eef_pos_{axis}' in self.data.columns for axis in ['x', 'y', 'z']):
            distances = []
            for i in range(1, len(self.data)):
                dist = np.sqrt(
                    (self.data['eef_pos_x'].iloc[i] - self.data['eef_pos_x'].iloc[i-1])**2 +
                    (self.data['eef_pos_y'].iloc[i] - self.data['eef_pos_y'].iloc[i-1])**2 +
                    (self.data['eef_pos_z'].iloc[i] - self.data['eef_pos_z'].iloc[i-1])**2
                )
                distances.append(dist)
            
            pos_stats['total_distance'] = sum(distances)
            pos_stats['avg_speed'] = pos_stats['total_distance'] / pos_stats['duration'] if pos_stats['duration'] > 0 else 0
        
        return pos_stats
    
    def create_trajectory_plot(self):
        """Create 3D trajectory plot with coordinate frame at origin"""
        if self.data is None:
            return None
        
        # Create 3D trajectory plot
        fig = go.Figure()
        
        # Add coordinate frame at origin
        origin = [0, 0, 0]
        axis_length = 0.1  # 10cm axes
        
        # X-axis (red)
        fig.add_trace(go.Scatter3d(
            x=[origin[0], origin[0] + axis_length],
            y=[origin[1], origin[1]],
            z=[origin[2], origin[2]],
            mode='lines',
            line=dict(color='red', width=8),
            name='X-axis',
            showlegend=True,
            hovertemplate='X-axis<extra></extra>'
        ))
        
        # Y-axis (green)
        fig.add_trace(go.Scatter3d(
            x=[origin[0], origin[0]],
            y=[origin[1], origin[1] + axis_length],
            z=[origin[2], origin[2]],
            mode='lines',
            line=dict(color='green', width=8),
            name='Y-axis',
            showlegend=True,
            hovertemplate='Y-axis<extra></extra>'
        ))
        
        # Z-axis (blue)
        fig.add_trace(go.Scatter3d(
            x=[origin[0], origin[0]],
            y=[origin[1], origin[1]],
            z=[origin[2], origin[2] + axis_length],
            mode='lines',
            line=dict(color='blue', width=8),
            name='Z-axis',
            showlegend=True,
            hovertemplate='Z-axis<extra></extra>'
        ))
        
        # Add origin point
        fig.add_trace(go.Scatter3d(
            x=[origin[0]],
            y=[origin[1]],
            z=[origin[2]],
            mode='markers',
            marker=dict(size=8, color='black', symbol='circle'),
            name='Origin',
            showlegend=True,
            hovertemplate='<b>Origin (0,0,0)</b><extra></extra>'
        ))
        
        # Add 3D trajectory line with colorbar positioned on the left
        fig.add_trace(go.Scatter3d(
            x=self.data['eef_pos_x'],
            y=self.data['eef_pos_y'],
            z=self.data['eef_pos_z'],
            mode='lines+markers',
            line=dict(
                color=self.data['time_seconds'],
                colorscale='Viridis',
                width=4,
                colorbar=dict(
                    title="Time (s)",
                    x=-0.15,  # Position colorbar on the left side (-0.15 means 15% to the left of the plot)
                    xanchor="right",  # Anchor the right side of the colorbar to the x position
                    thickness=15,  # Thickness of the colorbar
                    len=0.8,  # Length of the colorbar (80% of plot height)
                    yanchor="middle"  # Center the colorbar vertically
                )
            ),
            marker=dict(
                size=3,
                color=self.data['time_seconds'],
                colorscale='Viridis',
                opacity=0.7
            ),
            name='EEF Trajectory',
            hovertemplate='<b>End-Effector Position</b><br>' +
                         'X: %{x:.4f} m<br>' +
                         'Y: %{y:.4f} m<br>' +
                         'Z: %{z:.4f} m<br>' +
                         'Time: %{marker.color:.2f} s<br>' +
                         '<extra></extra>'
        ))
        
        # Add start point
        fig.add_trace(go.Scatter3d(
            x=[self.data['eef_pos_x'].iloc[0]],
            y=[self.data['eef_pos_y'].iloc[0]],
            z=[self.data['eef_pos_z'].iloc[0]],
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
            x=[self.data['eef_pos_x'].iloc[-1]],
            y=[self.data['eef_pos_y'].iloc[-1]],
            z=[self.data['eef_pos_z'].iloc[-1]],
            mode='markers',
            marker=dict(size=8, color='red', symbol='diamond'),
            name='End',
            hovertemplate='<b>End Position</b><br>' +
                         'X: %{x:.4f} m<br>' +
                         'Y: %{y:.4f} m<br>' +
                         'Z: %{z:.4f} m<br>' +
                         '<extra></extra>'
        ))
        
        # Update layout with adjusted margins to accommodate left-side colorbar
        fig.update_layout(
            title='End-Effector 3D Trajectory with Coordinate Frame',
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Z (m)',
                aspectmode='data',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            height=1000,
            showlegend=True,
            margin=dict(l=100, r=50, t=50, b=50)  # Increase left margin to make room for colorbar
        )
        
        return fig
    
    def create_position_time_plot(self):
        """Create position vs time plot"""
        if self.data is None:
            return None
    
        fig = go.Figure()
    
        colors = ['red', 'green', 'blue']
        axes = ['x', 'y', 'z']
    
        for i, axis in enumerate(axes):
            col = f'eef_pos_{axis}'
            if col in self.data.columns:
                fig.add_trace(go.Scatter(
                    x=self.data['time_seconds'],
                    y=self.data[col],
                    mode='lines',
                    name=f'Position {axis.upper()}',
                    line=dict(color=colors[i], width=2),
                    hovertemplate=f'<b>Position {axis.upper()}</b><br>' +
                                 'Time: %{x:.2f} s<br>' +
                                 f'{axis.upper()}: %{{y:.4f}} m<br>' +  # Fixed: double braces for literal braces
                                 '<extra></extra>'
                ))
    
        fig.update_layout(
            title='End-Effector Position vs Time',
            xaxis_title='Time (s)',
            yaxis_title='Position (m)',
            height=400,
            showlegend=True,
            hovermode='x unified'
        )
    
        return fig
    
    def create_quaternion_plot(self):
        """Create quaternion components vs time plot"""
        if self.data is None:
            return None
        
        fig = go.Figure()
        
        colors = ['red', 'green', 'blue', 'orange']
        components = ['x', 'y', 'z', 'w']
        
        for i, comp in enumerate(components):
            col = f'eef_quat_{comp}'
            if col in self.data.columns:
                fig.add_trace(go.Scatter(
                    x=self.data['time_seconds'],
                    y=self.data[col],
                    mode='lines',
                    name=f'q{comp}',
                    line=dict(color=colors[i], width=2),
                    hovertemplate=f'<b>Quaternion q{comp}</b><br>' +
                                 'Time: %{x:.2f} s<br>' +
                                 f'q{comp}: %{{y:.4f}}<br>' +  # Fixed: double braces
                                 '<extra></extra>'
                ))
        
        # Calculate and add quaternion magnitude
        if all(f'eef_quat_{c}' in self.data.columns for c in components):
            magnitude = np.sqrt(
                self.data['eef_quat_x']**2 + 
                self.data['eef_quat_y']**2 + 
                self.data['eef_quat_z']**2 + 
                self.data['eef_quat_w']**2
            )
            
            fig.add_trace(go.Scatter(
                x=self.data['time_seconds'],
                y=magnitude,
                mode='lines',
                name='|q| (magnitude)',
                line=dict(color='black', width=2, dash='dash'),
                hovertemplate='<b>Quaternion Magnitude</b><br>' +
                             'Time: %{x:.2f} s<br>' +
                             '|q|: %{y:.4f}<br>' +  # Fixed: no f-string needed here
                             '<extra></extra>'
            ))
        
        fig.update_layout(
            title='End-Effector Quaternion Components vs Time',
            xaxis_title='Time (s)',
            yaxis_title='Quaternion Component',
            height=400,
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
    
    def create_velocity_plot(self):
        """Create velocity plot"""
        if self.data is None or len(self.data) < 2:
            return None
        
        # Calculate velocities
        dt = np.diff(self.data['time_seconds'])
        
        velocities = {}
        for axis in ['x', 'y', 'z']:
            col = f'eef_pos_{axis}'
            if col in self.data.columns:
                dx = np.diff(self.data[col])
                velocities[axis] = dx / dt
        
        if not velocities:
            return None
        
        fig = go.Figure()
        
        colors = ['red', 'green', 'blue']
        axes = ['x', 'y', 'z']
        
        time_mid = self.data['time_seconds'].iloc[:-1] + dt/2  # Midpoint times
        
        for i, axis in enumerate(axes):
            if axis in velocities:
                fig.add_trace(go.Scatter(
                    x=time_mid,
                    y=velocities[axis],
                    mode='lines',
                    name=f'Velocity {axis.upper()}',
                    line=dict(color=colors[i], width=2),
                    hovertemplate=f'<b>Velocity {axis.upper()}</b><br>' +
                                 'Time: %{x:.2f} s<br>' +
                                 f'v_{axis}: %{{y:.4f}} m/s<br>' +  # Fixed: double braces
                                 '<extra></extra>'
                ))
        
        # Calculate speed (magnitude of velocity)
        if len(velocities) == 3:
            speed = np.sqrt(velocities['x']**2 + velocities['y']**2 + velocities['z']**2)
            fig.add_trace(go.Scatter(
                x=time_mid,
                y=speed,
                mode='lines',
                name='Speed (|v|)',
                line=dict(color='black', width=3),
                hovertemplate='<b>Speed</b><br>' +
                             'Time: %{x:.2f} s<br>' +
                             '|v|: %{y:.4f} m/s<br>' +  # Fixed: no f-string needed
                             '<extra></extra>'
            ))
        
        fig.update_layout(
            title='End-Effector Velocity vs Time',
            xaxis_title='Time (s)',
            yaxis_title='Velocity (m/s)',
            height=400,
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
    
    def create_combined_dashboard(self):
        """Create a combined dashboard with all plots"""
        if self.data is None:
            return None
        
        # Create subplots with better proportions for 3D plot
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('EE-Position vs Time', '3D Trajectory with Coordinate Frame', 
                          'EE-Quaternion vs Time', 'EE-Velocity vs Time'),
            specs=[[{'type': 'scatter'}, {'type': 'scatter3d'}],
                   [{'type': 'scatter'}, {'type': 'scatter'}]],
            vertical_spacing=0.15,
            horizontal_spacing=0.05,
            column_widths=[0.45, 0.55],  # More space for 3D plot
            row_heights=[0.55, 0.45]     # More space for top row
        )
        
        # Add position vs time (row 1, col 1)
        colors = ['red', 'green', 'blue']
        for i, axis in enumerate(['x', 'y', 'z']):
            col = f'eef_pos_{axis}'
            if col in self.data.columns:
                fig.add_trace(go.Scatter(
                    x=self.data['time_seconds'],
                    y=self.data[col],
                    mode='lines',
                    name=f'Pos {axis.upper()}',
                    line=dict(color=colors[i], width=2),
                    legendgroup='position'
                ), row=1, col=1)
    
        # Calculate adaptive coordinate frame size
        pos_ranges = {
            'x': self.data['eef_pos_x'].max() - self.data['eef_pos_x'].min(),
            'y': self.data['eef_pos_y'].max() - self.data['eef_pos_y'].min(),
            'z': self.data['eef_pos_z'].max() - self.data['eef_pos_z'].min()
        }
        max_range = max(pos_ranges.values())
        axis_length = max(0.08, min(0.25, max_range * 0.4))  # 40% of trajectory range
    
        # Add coordinate frame to 3D plot (row 1, col 2)
        origin = [0, 0, 0]
        
        # X-axis (red)
        fig.add_trace(go.Scatter3d(
            x=[origin[0], origin[0] + axis_length],
            y=[origin[1], origin[1]],
            z=[origin[2], origin[2]],
            mode='lines',
            line=dict(color='red', width=8),  # Thicker lines
            name='X-axis',
            showlegend=False
        ), row=1, col=2)
        
        # Y-axis (green)
        fig.add_trace(go.Scatter3d(
            x=[origin[0], origin[0]],
            y=[origin[1], origin[1] + axis_length],
            z=[origin[2], origin[2]],
            mode='lines',
            line=dict(color='green', width=8),  # Thicker lines
            name='Y-axis',
            showlegend=False
        ), row=1, col=2)
        
        # Z-axis (blue)
        fig.add_trace(go.Scatter3d(
            x=[origin[0], origin[0]],
            y=[origin[1], origin[1]],
            z=[origin[2], origin[2] + axis_length],
            mode='lines',
            line=dict(color='blue', width=8),  # Thicker lines
            name='Z-axis',
            showlegend=False
        ), row=1, col=2)
        
        # Add origin point
        fig.add_trace(go.Scatter3d(
            x=[origin[0]],
            y=[origin[1]],
            z=[origin[2]],
            mode='markers',
            marker=dict(size=10, color='black', symbol='circle'),  # Larger marker
            name='Origin',
            showlegend=False
        ), row=1, col=2)
        
        # Add 3D trajectory (row 1, col 2)
        fig.add_trace(go.Scatter3d(
            x=self.data['eef_pos_x'],
            y=self.data['eef_pos_y'],
            z=self.data['eef_pos_z'],
            mode='lines+markers',
            line=dict(color=self.data['time_seconds'], colorscale='Viridis', width=4),
            marker=dict(size=3, color=self.data['time_seconds'], colorscale='Viridis'),
            name='Trajectory',
            showlegend=False
        ), row=1, col=2)
        
        # Add quaternion vs time (moved to row 2, col 1)
        quat_colors = ['red', 'green', 'blue', 'orange']
        for i, comp in enumerate(['x', 'y', 'z', 'w']):
            col = f'eef_quat_{comp}'
            if col in self.data.columns:
                fig.add_trace(go.Scatter(
                    x=self.data['time_seconds'],
                    y=self.data[col],
                    mode='lines',
                    name=f'q{comp}',
                    line=dict(color=quat_colors[i], width=2),
                    legendgroup='quaternion'
                ), row=2, col=1)
    
        # Add velocity vs time (stays at row 2, col 2)
        if len(self.data) > 1:
            dt = np.diff(self.data['time_seconds'])
            time_mid = self.data['time_seconds'].iloc[:-1] + dt/2
            
            for i, axis in enumerate(['x', 'y', 'z']):
                col = f'eef_pos_{axis}'
                if col in self.data.columns:
                    velocities = np.diff(self.data[col]) / dt
                    fig.add_trace(go.Scatter(
                        x=time_mid,
                        y=velocities,
                        mode='lines',
                        name=f'Vel {axis.upper()}',
                        line=dict(color=colors[i], width=2),
                        legendgroup='velocity'
                    ), row=2, col=2)
    
        # Update layout
        fig.update_layout(
            height=900,  # Increased height
            title_text=f"End-Effector Dynamics Dashboard - {os.path.basename(self.csv_file)}",
            showlegend=True
        )
        
        # Update 3D scene with better scaling
        fig.update_scenes(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='cube',  # Better aspect ratio
            camera=dict(
                eye=dict(x=1.3, y=1.3, z=1.3),  # Better camera position
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0)
            ),
            # Set explicit ranges with padding
            xaxis=dict(
                range=[min(0, self.data['eef_pos_x'].min()-0.1), 
                       max(axis_length, self.data['eef_pos_x'].max()+0.1)]
            ),
            yaxis=dict(
                range=[min(0, self.data['eef_pos_y'].min()-0.1), 
                       max(axis_length, self.data['eef_pos_y'].max()+0.1)]
            ),
            zaxis=dict(
                range=[min(0, self.data['eef_pos_z'].min()-0.1), 
                       max(axis_length, self.data['eef_pos_z'].max()+0.1)]
            )
        )
        
        # Update axis labels - Updated positions
        fig.update_xaxes(title_text="Time (s)", row=1, col=1)  # Position plot
        fig.update_yaxes(title_text="Position (m)", row=1, col=1)
        
        fig.update_xaxes(title_text="Time (s)", row=2, col=1)  # Quaternion plot
        fig.update_yaxes(title_text="Quaternion", row=2, col=1)
        
        fig.update_xaxes(title_text="Time (s)", row=2, col=2)  # Velocity plot
        fig.update_yaxes(title_text="Velocity (m/s)", row=2, col=2)
        
        return fig
    
    def print_statistics(self):
        """Print trajectory statistics"""
        if self.data is None:
            return
        
        stats = self.calculate_trajectory_stats()
        
        print("\n" + "="*70)
        print("TRAJECTORY STATISTICS")
        print("="*70)
        print(f"File: {os.path.basename(self.csv_file)}")
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
        if all(f'eef_quat_{c}' in self.data.columns for c in ['x', 'y', 'z', 'w']):
            magnitude = np.sqrt(
                self.data['eef_quat_x']**2 + 
                self.data['eef_quat_y']**2 + 
                self.data['eef_quat_z']**2 + 
                self.data['eef_quat_w']**2
            )
            print(f"\nQuaternion Statistics:")
            print(f"  Magnitude range: [{magnitude.min():.6f}, {magnitude.max():.6f}]")
            print(f"  Magnitude mean: {magnitude.mean():.6f}")
            print(f"  Magnitude std: {magnitude.std():.6f}")
        
        print("="*70)
    
    def plot_all(self, show_stats=True, save_html=None, show_plots=True):
        """Plot all visualizations"""
        if not self.load_data():
            return False
        
        if show_stats:
            self.print_statistics()
        
        # Create individual plots
        print("\n📊 Creating plots...")
        
        # 1. 3D Trajectory Plot
        print("  - 3D Trajectory")
        traj_fig = self.create_trajectory_plot()
        
        # 2. Position vs Time Plot
        print("  - Position vs Time")
        pos_fig = self.create_position_time_plot()
        
        # 3. Quaternion vs Time Plot
        print("  - Quaternion vs Time") 
        quat_fig = self.create_quaternion_plot()
        
        # 4. Velocity vs Time Plot
        print("  - Velocity vs Time")
        vel_fig = self.create_velocity_plot()
        
        # 5. Combined Dashboard
        print("  - Combined Dashboard")
        dashboard_fig = self.create_combined_dashboard()
        
        # Save HTML files if requested
        if save_html:
            print(f"\n💾 Saving HTML files...")
            base_name = save_html.replace('.html', '')
            
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
            print(f"\n🖥️ Displaying plots...")
            
            if dashboard_fig:
                print("  - Opening Combined Dashboard")
                dashboard_fig.show()
            
            if traj_fig:
                print("  - Opening 3D Trajectory")
                traj_fig.show()
        
        return True


def main():
    parser = argparse.ArgumentParser(description="Plot End-Effector Trajectory and Quaternion Data")
    parser.add_argument("--csv", type=str, default=None,
                       help="Path to CSV file (if not provided, will find latest)")
    parser.add_argument("--no-stats", action="store_true",
                       help="Don't print statistics")
    parser.add_argument("--save-html", type=str, default=None,
                       help="Save plots as HTML files (provide base filename)")
    parser.add_argument("--no-show", action="store_true",
                       help="Don't show interactive plots")
    parser.add_argument("--data-dir", type=str, default=None,
                       help="Directory to search for CSV files")
    
    args = parser.parse_args()
    
    try:
        print("🎯 EEF Trajectory and Quaternion Plotter")
        print("="*50)
        
        # Create plotter
        plotter = EEFTrajectoryPlotter(csv_file=args.csv)
        
        # Override data directory if provided
        if args.data_dir and args.csv is None:
            plotter.csv_file = plotter.find_latest_csv(args.data_dir)
        
        # Plot all visualizations
        success = plotter.plot_all(
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