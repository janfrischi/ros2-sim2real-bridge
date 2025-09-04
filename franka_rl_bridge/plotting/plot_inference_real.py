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
    def __init__(self, csv_file=None, trial_id=None):
        self.csv_file = csv_file
        self.data = None
        self.trial_id = trial_id
        self.config_description = None
        # Thesis style constants
        self._title_font_size = 14
        self._axis_font_size = 12
        self._legend_font_size = 12
        self._font_family = "Arial"   # or "DejaVu Sans" to match Matplotlib default

    # ---------------------- NEW: unified styling helper ----------------------
    def _apply_thesis_style(self, fig, title=None, x_title=None, y_title=None, height=None):
        """Apply consistent thesis styling to a 2D Plotly figure."""
        layout_updates = {
            "font": dict(family=self._font_family, size=self._axis_font_size),
            "plot_bgcolor": "white",
            "paper_bgcolor": "white",
            "hovermode": "x unified",
            "margin": dict(l=70, r=30, t=60, b=60),
            "legend": dict(
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="rgba(0,0,0,0.15)",
                borderwidth=1,
                font=dict(size=self._legend_font_size)
            ),
        }
        if title is not None:
            layout_updates["title"] = dict(
                text=title,
                font=dict(size=self._title_font_size),
                x=0.5,
                xanchor="center"
            )
        if height is not None:
            layout_updates["height"] = height

        fig.update_layout(**layout_updates)

        # Apply axis formatting (only affects 2D axes)
        fig.update_xaxes(
            title_text=x_title,
            title_font=dict(size=self._axis_font_size),
            showgrid=True,
            gridcolor="rgba(0,0,0,0.3)",
            zeroline=False
        )
        fig.update_yaxes(
            title_text=y_title,
            title_font=dict(size=self._axis_font_size),
            showgrid=True,
            gridcolor="rgba(0,0,0,0.3)",
            zeroline=False
        )
        return fig
    # -------------------------------------------------------------------------

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
            
            # Filter by trial if trial_id is specified
            if self.trial_id is not None:
                if 'trial_id' not in self.data.columns:
                    print("❌ 'trial_id' column not found in CSV.")
                    return False
                if self.trial_id not in self.data['trial_id'].unique():
                    print(f"❌ Trial {self.trial_id} not found in CSV.")
                    return False
                self.data = self.data[self.data['trial_id'] == self.trial_id].reset_index(drop=True)
                print(f"✅ Filtered to trial {self.trial_id}: {len(self.data)} data points")
            
            # Extract config_description from the first row of the filtered data
            if 'config_name' in self.data.columns and len(self.data) > 0:
                self.config_description = self.data.iloc[0]['config_name']
            else:
                self.config_description = None
            
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
            title=dict(text='End-Effector 3D Trajectory with Coordinate Frame',
                       font=dict(size=self._title_font_size),
                       x=0.5, xanchor='center'),
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Z (m)',
                xaxis=dict(backgroundcolor="white"),
                yaxis=dict(backgroundcolor="white"),
                zaxis=dict(backgroundcolor="white"),
            ),
            font=dict(family=self._font_family, size=self._axis_font_size),
            height=900,
            showlegend=True,
            margin=dict(l=100, r=50, t=60, b=50),
            paper_bgcolor='white'
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
    
        fig = self._apply_thesis_style(
            fig,
            title="End-Effector Position vs Time",
            x_title="Time (s)",
            y_title="Position (m)",
            height=400
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
        
        fig = self._apply_thesis_style(
            fig,
            title="End-Effector Quaternion Components vs Time",
            x_title="Time (s)",
            y_title="Quaternion Component",
            height=400
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
        
        fig = self._apply_thesis_style(
            fig,
            title="End-Effector Velocity vs Time",
            x_title="Time (s)",
            y_title="Velocity (m/s)",
            height=400
        )
        
        return fig
    
    def create_manipulability_plot(self):
        """Create manipulability index vs time plot"""
        if self.data is None:
            return None
        
        # Check if manipulability_index column exists
        if 'manipulability_index' not in self.data.columns:
            print("⚠️ No manipulability_index column found in data")
            return None
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=self.data['time_seconds'],
            y=self.data['manipulability_index'],
            mode='lines',
            name='Manipulability Index',
            line=dict(color='purple', width=3),
            hovertemplate='<b>Manipulability Index</b><br>' +
                         'Time: %{x:.2f} s<br>' +
                         'Index: %{y:.6f}<br>' +
                         '<extra></extra>'
        ))
        
        # Add mean line
        mean_value = self.data['manipulability_index'].mean()
        fig.add_hline(
            y=mean_value,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {mean_value:.6f}",
            annotation_position="top right"
        )
        
        fig = self._apply_thesis_style(
            fig,
            title="Robot Manipulability Index vs Time",
            x_title="Time (s)",
            y_title="Manipulability Index",
            height=400
        )
        
        return fig
    
    def create_gripper_plot(self):
        """Create gripper command vs time plot"""
        if self.data is None:
            return None
    
        # Check if action_gripper column exists
        if 'action_gripper' not in self.data.columns:
            print("⚠️ No action_gripper column found in data")
            return None
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=self.data['time_seconds'],
            y=self.data['action_gripper'],
            mode='lines+markers',
            name='Gripper Command',
            line=dict(color='orange', width=3),
            marker=dict(size=4, color='orange'),
            hovertemplate='<b>Gripper Command</b><br>' +
                         'Time: %{x:.2f} s<br>' +
                         'Command: %{y:.3f}<br>' +
                         '<extra></extra>'
        ))
        
        # Add horizontal reference lines for open/closed states
        fig.add_hline(
            y=1.0,
            line_dash="dash",
            line_color="green",
            annotation_text="Open (1.0)",
            annotation_position="top right"
        )
        
        fig.add_hline(
            y=-1.0,
            line_dash="dash", 
            line_color="red",
            annotation_text="Closed (-1.0)",
            annotation_position="bottom right"
        )
        
        fig.add_hline(
            y=0.0,
            line_dash="dot",
            line_color="gray",
            annotation_text="Neutral (0.0)",
            annotation_position="bottom left"  # Changed from "middle right" to "bottom left"
        )
        
        fig = self._apply_thesis_style(
            fig,
            title="Gripper Command vs Time",
            x_title="Time (s)",
            y_title="Gripper Command",
            height=400
        )
        # Preserve custom y-range
        fig.update_yaxes(range=[-1.2, 1.2])
        
        return fig
    
    def create_combined_dashboard(self):
        """Create a combined dashboard with all plots"""
        if self.data is None:
            return None
        
        # Create subplots with 2x3 layout to include gripper plot
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('EE-Position vs Time', '3D Trajectory with Coordinate Frame', 'Gripper Command vs Time',
                          'EE-Quaternion vs Time', 'Manipulability Index vs Time', 'Trajectory Statistics'),
            specs=[[{'type': 'scatter'}, {'type': 'scatter3d'}, {'type': 'scatter'}],
                   [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}]],
            vertical_spacing=0.12,
            horizontal_spacing=0.08,
            column_widths=[0.33, 0.34, 0.33],
            row_heights=[0.55, 0.45]
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
        axis_length = max(0.08, min(0.25, max_range * 0.4))

        # Add coordinate frame to 3D plot (row 1, col 2)
        origin = [0, 0, 0]
        
        # X-axis (red)
        fig.add_trace(go.Scatter3d(
            x=[origin[0], origin[0] + axis_length],
            y=[origin[1], origin[1]],
            z=[origin[2], origin[2]],
            mode='lines',
            line=dict(color='red', width=8),
            name='X-axis',
            showlegend=False
        ), row=1, col=2)
        
        # Y-axis (green)
        fig.add_trace(go.Scatter3d(
            x=[origin[0], origin[0]],
            y=[origin[1], origin[1] + axis_length],
            z=[origin[2], origin[2]],
            mode='lines',
            line=dict(color='green', width=8),
            name='Y-axis',
            showlegend=False
        ), row=1, col=2)
        
        # Z-axis (blue)
        fig.add_trace(go.Scatter3d(
            x=[origin[0], origin[0]],
            y=[origin[1], origin[1]],
            z=[origin[2], origin[2] + axis_length],
            mode='lines',
            line=dict(color='blue', width=8),
            name='Z-axis',
            showlegend=False
        ), row=1, col=2)
        
        # Add origin point
        fig.add_trace(go.Scatter3d(
            x=[origin[0]],
            y=[origin[1]],
            z=[origin[2]],
            mode='markers',
            marker=dict(size=10, color='black', symbol='circle'),
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
        
        # Add gripper command vs time (row 1, col 3)
        if 'action_gripper' in self.data.columns:
            fig.add_trace(go.Scatter(
                x=self.data['time_seconds'],
                y=self.data['action_gripper'],
                mode='lines+markers',
                name='Gripper Cmd',
                line=dict(color='orange', width=2),
                marker=dict(size=3, color='orange'),
                legendgroup='gripper'
            ), row=1, col=3)
            
            # Add reference lines for gripper states using scatter traces
            time_range = [self.data['time_seconds'].min(), self.data['time_seconds'].max()]
            
            # Open line (green, y=1.0)
            fig.add_trace(go.Scatter(
                x=time_range,
                y=[1.0, 1.0],
                mode='lines',
                name='Open (1.0)',
                line=dict(color='green', width=2, dash='dash'),
                showlegend=False
            ), row=1, col=3)
            
            # Closed line (red, y=-1.0)
            fig.add_trace(go.Scatter(
                x=time_range,
                y=[-1.0, -1.0],
                mode='lines',
                name='Closed (-1.0)',
                line=dict(color='red', width=2, dash='dash'),
                showlegend=False
            ), row=1, col=3)
            
            # Neutral line (gray, y=0.0)
            fig.add_trace(go.Scatter(
                x=time_range,
                y=[0.0, 0.0],
                mode='lines',
                name='Neutral (0.0)',
                line=dict(color='gray', width=2, dash='dot'),
                showlegend=False
            ), row=1, col=3)
    
        # Add quaternion vs time (row 2, col 1)
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

        # Add manipulability index vs time (row 2, col 2)
        if 'manipulability_index' in self.data.columns:
            fig.add_trace(go.Scatter(
                x=self.data['time_seconds'],
                y=self.data['manipulability_index'],
                mode='lines',
                name='Manipulability',
                line=dict(color='purple', width=3),
                legendgroup='manipulability'
            ), row=2, col=2)
            
            # Add mean line using scatter trace
            mean_value = self.data['manipulability_index'].mean()
            time_range = [self.data['time_seconds'].min(), self.data['time_seconds'].max()]
            fig.add_trace(go.Scatter(
                x=time_range,
                y=[mean_value, mean_value],
                mode='lines',
                name=f'Mean ({mean_value:.6f})',
                line=dict(color='red', width=2, dash='dash'),
                legendgroup='manipulability',
                showlegend=False
            ), row=2, col=2)

        # Update layout
        title_text = f"End-Effector Dynamics Dashboard - {os.path.basename(self.csv_file)}"
        if self.trial_id is not None:
            title_text += f" (Trial {self.trial_id})"
        if self.config_description:
            title_text += f" - {self.config_description}"
        
        fig.update_layout(
            font=dict(family=self._font_family, size=self._axis_font_size),
            title=dict(text=title_text, font=dict(size=self._title_font_size), x=0.5, xanchor='center'),
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="rgba(0,0,0,0.15)",
                borderwidth=1,
                font=dict(size=self._legend_font_size),
            )
        )
        
        # Uniform axis styling for all 2D subplots
        for r in [1, 2]:
            for c in [1, 3] if r == 1 else [1, 2]:
                fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.3)", zeroline=False, row=r, col=c,
                                 title_font=dict(size=self._axis_font_size))
                fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.3)", zeroline=False, row=r, col=c,
                                 title_font=dict(size=self._axis_font_size))
        
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
        
        # Gripper statistics
        if 'action_gripper' in self.data.columns:
            gripper_data = self.data['action_gripper']
            print(f"\nGripper Command Statistics:")
            print(f"  Range: [{gripper_data.min():.3f}, {gripper_data.max():.3f}]")
            print(f"  Mean: {gripper_data.mean():.3f}")
            print(f"  Std: {gripper_data.std():.3f}")
            
            # Count command states
            open_commands = (gripper_data > 0.5).sum()
            closed_commands = (gripper_data < -0.5).sum()
            neutral_commands = ((gripper_data >= -0.5) & (gripper_data <= 0.5)).sum()
            
            print(f"  Command Distribution:")
            print(f"    Open (>0.5): {open_commands} ({open_commands/len(gripper_data)*100:.1f}%)")
            print(f"    Closed (<-0.5): {closed_commands} ({closed_commands/len(gripper_data)*100:.1f}%)")
            print(f"    Neutral (-0.5 to 0.5): {neutral_commands} ({neutral_commands/len(gripper_data)*100:.1f}%)")
        
        # Manipulability statistics
        if 'manipulability_index' in self.data.columns:
            manip_data = self.data['manipulability_index']
            print(f"\nManipulability Index Statistics:")
            print(f"  Range: [{manip_data.min():.6f}, {manip_data.max():.6f}]")
            print(f"  Mean: {manip_data.mean():.6f}")
            print(f"  Std: {manip_data.std():.6f}")
            print(f"  Median: {manip_data.median():.6f}")
        
        print("="*70)
    
    # Update the plot_all method to include PDF export
    def plot_all(self, show_stats=True, save_html=None, show_plots=True):
        """Plot all visualizations and optionally save HTML + PDF"""
        if not self.load_data():
            return False
        
        if show_stats:
            self.print_statistics()
        
        # Create individual plots
        print("\n📊 Creating plots...")
        traj_fig = self.create_trajectory_plot()
        pos_fig = self.create_position_time_plot()
        quat_fig = self.create_quaternion_plot()
        gripper_fig = self.create_gripper_plot()
        manip_fig = self.create_manipulability_plot()
        dashboard_fig = self.create_combined_dashboard()
        
        # Save HTML (existing behavior) + new PDF export
        if save_html:
            print(f"\n💾 Saving HTML + PDF files...")
            base_name = save_html.replace('.html', '')
            
            figures = [
                ('trajectory', traj_fig),
                ('position', pos_fig),
                ('quaternion', quat_fig),
                ('gripper', gripper_fig),
                ('manipulability', manip_fig),
                ('dashboard', dashboard_fig),
            ]
            
            # Common image export config (fine for PDF vector output)
            pdf_cfg = dict(width=1200, height=800, scale=2, format='pdf')
            
            for suffix, fig in figures:
                if fig:
                    html_path = f"{base_name}_{suffix}.html"
                    pdf_path = f"{base_name}_{suffix}.pdf"
                    try:
                        fig.write_html(html_path)
                        print(f"  ✅ {html_path}")
                    except Exception as e:
                        print(f"  ⚠️ Failed HTML ({suffix}): {e}")
                    try:
                        fig.write_image(pdf_path, **pdf_cfg)
                        print(f"    📄 PDF saved: {pdf_path}")
                    except Exception as e:
                        print(f"    ⚠️ Failed PDF ({suffix}): {e} (install with: pip install kaleido)")
        
        if show_plots:
            print(f"\n🖥️ Displaying plots...")
            if dashboard_fig:
                print("  - Opening Combined Dashboard")
                dashboard_fig.show()
            if traj_fig:
                print("  - Opening 3D Trajectory")
                traj_fig.show()
            if gripper_fig:
                print("  - Opening Gripper Command")
                gripper_fig.show()
            if manip_fig:
                print("  - Opening Manipulability Index")
                manip_fig.show()
        
        return True

def main():
    parser = argparse.ArgumentParser(description="Plot End-Effector Trajectory and Quaternion Data")
    parser.add_argument("--csv", type=str, default=None,
                       help="Path to CSV file (if not provided, will find latest)")
    parser.add_argument("--trial", type=int, default=None,
                       help="Specific trial ID to plot (if not provided, plots all data)")
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
        
        # Create plotter with trial_id
        plotter = EEFTrajectoryPlotter(csv_file=args.csv, trial_id=args.trial)
        
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