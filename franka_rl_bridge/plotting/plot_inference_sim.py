#!/usr/bin/env python3
"""
Visualize observation from inference in simulation -> observation.csv can be found in the Isaaclab Project robomimic/data/observation.csv
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def smooth_series(series, window=5):
    """Simple moving average smoothing"""
    return series.rolling(window=window, center=True, min_periods=1).mean()

class ObservationsPlotter:
    def __init__(self, csv_file: str, trial_id: int = None):
        self.csv_file = csv_file
        self.data = None
        self.trial_id = trial_id
        self.config_description = None

    def load_data(self) -> bool:
        """Load observation data from CSV file and filter by trial if specified"""
        try:
            df = pd.read_csv(self.csv_file)
            if self.trial_id is not None:
                if 'trial' not in df.columns:
                    print("❌ 'trial' column not found in CSV.")
                    return False
                if self.trial_id not in df['trial'].unique():
                    print(f"❌ Trial {self.trial_id} not found in CSV.")
                    return False
                df = df[df['trial'] == self.trial_id].reset_index(drop=True)
            self.data = df
            # Extract config_description from the first row of the filtered data
            if 'config_description' in df.columns and len(df) > 0:
                self.config_description = df.iloc[0]['config_description']
            else:
                self.config_description = None
            return True
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            return False

    def create_3d_eef_trajectory_plot(self, smooth_window=5) -> go.Figure:
        """Create 3D trajectory plot of the end-effector with smoothing"""
        x = smooth_series(self.data['eef_pos_x'], window=smooth_window)
        y = smooth_series(self.data['eef_pos_y'], window=smooth_window)
        z = smooth_series(self.data['eef_pos_z'], window=smooth_window)

        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='lines+markers',
            line=dict(color='blue', width=4),
            marker=dict(size=3, color='blue'),
            name='EEF Trajectory (Smoothed)',
            hovertemplate='<b>EEF Position</b><br>' +
                         'X: %{x:.4f} m<br>' +
                         'Y: %{y:.4f} m<br>' +
                         'Z: %{z:.4f} m<br>' +
                         '<extra></extra>'
        ))
        fig.add_trace(go.Scatter3d(
            x=[x.iloc[0]],
            y=[y.iloc[0]],
            z=[z.iloc[0]],
            mode='markers',
            marker=dict(size=8, color='green', symbol='diamond'),
            name='Start',
            hovertemplate='<b>Start Position</b><br>' +
                         'X: %{x:.4f} m<br>' +
                         'Y: %{y:.4f} m<br>' +
                         'Z: %{z:.4f} m<br>' +
                         '<extra></extra>'
        ))
        fig.add_trace(go.Scatter3d(
            x=[x.iloc[-1]],
            y=[y.iloc[-1]],
            z=[z.iloc[-1]],
            mode='markers',
            marker=dict(size=8, color='red', symbol='diamond'),
            name='End',
            hovertemplate='<b>End Position</b><br>' +
                         'X: %{x:.4f} m<br>' +
                         'Y: %{y:.4f} m<br>' +
                         'Z: %{z:.4f} m<br>' +
                         '<extra></extra>'
        ))
        # Set the plot title using config_description
        if self.config_description:
            fig.update_layout(title_text=self.config_description)
        else:
            fig.update_layout(title_text="End-Effector Trajectory")
        return fig

    def create_position_time_plot(self, smooth_window=5) -> go.Figure:
        """Create position vs time plot with smoothing"""
        fig = go.Figure()
        colors = ['red', 'green', 'blue']
        axes = ['x', 'y', 'z']
        for i, axis in enumerate(axes):
            fig.add_trace(go.Scatter(
                x=self.data['timestamp'],
                y=smooth_series(self.data[f'eef_pos_{axis}'], window=smooth_window),
                mode='lines',
                name=f'EEF Pos {axis.upper()} (Smoothed)',
                line=dict(color=colors[i], width=2),
                hovertemplate=f'<b>EEF Pos {axis.upper()}</b><br>' +
                              'Time: %{x:.2f} s<br>' +
                              f'{axis.upper()}: %{{y:.4f}} m<br>' +
                              '<extra></extra>'
            ))
        fig.update_layout(
            title='EEF Position vs Time',
            xaxis_title='Time (s)',
            yaxis_title='Position (m)',
            height=400,
            showlegend=True,
            hovermode='x unified'
        )
        return fig

    def create_quaternion_time_plot(self, smooth_window=5) -> go.Figure:
        """Create quaternion components vs time plot with smoothing"""
        fig = go.Figure()
        colors = ['red', 'green', 'blue', 'orange']
        components = ['x', 'y', 'z', 'w']
        for i, comp in enumerate(components):
            fig.add_trace(go.Scatter(
                x=self.data['timestamp'],
                y=smooth_series(self.data[f'eef_quat_{comp}'], window=smooth_window),
                mode='lines',
                name=f'EEF q{comp} (Smoothed)',
                line=dict(color=colors[i], width=2),
                hovertemplate=f'<b>EEF q{comp}</b><br>' +
                              'Time: %{x:.2f} s<br>' +
                              f'q{comp}: %{{y:.4f}}<br>' +
                              '<extra></extra>'
            ))
        quat_mag = np.sqrt(
            smooth_series(self.data['eef_quat_x'], window=smooth_window)**2 +
            smooth_series(self.data['eef_quat_y'], window=smooth_window)**2 +
            smooth_series(self.data['eef_quat_z'], window=smooth_window)**2 +
            smooth_series(self.data['eef_quat_w'], window=smooth_window)**2
        )
        fig.add_trace(go.Scatter(
            x=self.data['timestamp'],
            y=quat_mag,
            mode='lines',
            name='|q| Magnitude (Smoothed)',
            line=dict(color='black', width=2, dash='dash'),
            hovertemplate='<b>Quaternion Magnitude</b><br>' +
                          'Time: %{x:.2f} s<br>' +
                          '|q|: %{y:.4f}<br>' +
                          '<extra></extra>'
        ))
        fig.update_layout(
            title='EEF Quaternion Components vs Time',
            xaxis_title='Time (s)',
            yaxis_title='Quaternion',
            height=400,
            showlegend=True,
            hovermode='x unified'
        )
        return fig

    def create_gripper_plot(self, smooth_window=5) -> go.Figure:
        """Plot gripper position or state vs time with smoothing"""
        fig = go.Figure()
        if 'gripper_pos' in self.data.columns:
            fig.add_trace(go.Scatter(
                x=self.data['timestamp'],
                y=smooth_series(self.data['gripper_pos'], window=smooth_window),
                mode='lines',
                name='Gripper Pos (Smoothed)',
                line=dict(color='purple', width=2),
                hovertemplate='<b>Gripper Pos</b><br>' +
                              'Time: %{x:.2f} s<br>' +
                              'Pos: %{y:.4f}<br>' +
                              '<extra></extra>'
            ))
        fig.update_layout(
            title='Gripper Position vs Time',
            xaxis_title='Time (s)',
            yaxis_title='Gripper Position',
            height=300,
            showlegend=True,
            hovermode='x unified'
        )
        return fig

    def create_cube_positions_plot(self, smooth_window=5) -> go.Figure:
        """Plot cube positions vs time with smoothing"""
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            subplot_titles=('Cube 1 Position', 'Cube 2 Position', 'Cube 3 Position'))
        colors = ['blue', 'red', 'green']
        for idx, cube in enumerate(['cube_1', 'cube_2', 'cube_3']):
            for i, axis in enumerate(['x', 'y', 'z']):
                fig.add_trace(go.Scatter(
                    x=self.data['timestamp'],
                    y=smooth_series(self.data[f'{cube}_pos_{axis}'], window=smooth_window),
                    mode='lines',
                    name=f'{cube} {axis.upper()} (Smoothed)',
                    line=dict(color=colors[idx], width=2, dash=['solid', 'dot', 'dash'][i]),
                    hovertemplate=f'<b>{cube} {axis.upper()}</b><br>' +
                                  'Time: %{x:.2f} s<br>' +
                                  f'{axis.upper()}: %{{y:.4f}} m<br>' +
                                  '<extra></extra>'
                ), row=idx+1, col=1)
        fig.update_layout(
            title='Cube Positions vs Time',
            xaxis_title='Time (s)',
            height=700,
            showlegend=True,
            hovermode='x unified'
        )
        return fig

    def create_combined_dashboard(self, smooth_window=5) -> go.Figure:
        """Create a combined dashboard with all plots (smoothed)"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('EEF Position vs Time', '3D EEF Trajectory',
                            'EEF Quaternion vs Time', 'Cube Positions'),
            specs=[[{'type': 'scatter'}, {'type': 'scatter3d'}],
                   [{'type': 'scatter'}, {'type': 'scatter'}]],
            vertical_spacing=0.12,
            horizontal_spacing=0.08
        )
        # Position vs time
        colors = ['red', 'green', 'blue']
        axes = ['x', 'y', 'z']
        for i, axis in enumerate(axes):
            fig.add_trace(go.Scatter(
                x=self.data['timestamp'],
                y=smooth_series(self.data[f'eef_pos_{axis}'], window=smooth_window),
                mode='lines',
                name=f'EEF Pos {axis.upper()} (Smoothed)',
                line=dict(color=colors[i], width=2),
            ), row=1, col=1)
        # 3D trajectory
        fig.add_trace(go.Scatter3d(
            x=smooth_series(self.data['eef_pos_x'], window=smooth_window),
            y=smooth_series(self.data['eef_pos_y'], window=smooth_window),
            z=smooth_series(self.data['eef_pos_z'], window=smooth_window),
            mode='lines+markers',
            line=dict(color='blue', width=3),
            marker=dict(size=2, color='blue'),
            name='EEF Trajectory (Smoothed)'
        ), row=1, col=2)
        # Quaternion vs time
        quat_colors = ['red', 'green', 'blue', 'orange']
        components = ['x', 'y', 'z', 'w']
        for i, comp in enumerate(components):
            fig.add_trace(go.Scatter(
                x=self.data['timestamp'],
                y=smooth_series(self.data[f'eef_quat_{comp}'], window=smooth_window),
                mode='lines',
                name=f'EEF q{comp} (Smoothed)',
                line=dict(color=quat_colors[i], width=2),
            ), row=2, col=1)
        # Cube 1 position vs time (as example)
        for axis in axes:
            fig.add_trace(go.Scatter(
                x=self.data['timestamp'],
                y=smooth_series(self.data[f'cube_1_pos_{axis}'], window=smooth_window),
                mode='lines',
                name=f'Cube 1 {axis.upper()} (Smoothed)',
                line=dict(color='blue', width=2, dash=['solid', 'dot', 'dash'][axes.index(axis)]),
            ), row=2, col=2)
        fig.update_layout(
            height=900,
            title_text=f"Observations Dashboard - {os.path.basename(self.csv_file)}",
            showlegend=True
        )
        # Update axis labels
        fig.update_xaxes(title_text="Time (s)", row=1, col=1)
        fig.update_yaxes(title_text="Position (m)", row=1, col=1)
        fig.update_xaxes(title_text="Time (s)", row=2, col=1)
        fig.update_yaxes(title_text="Quaternion", row=2, col=1)
        fig.update_xaxes(title_text="Time (s)", row=2, col=2)
        fig.update_yaxes(title_text="Cube 1 Position (m)", row=2, col=2)
        return fig

    def print_statistics(self):
        """Print basic statistics"""
        print("\n" + "="*60)
        print("OBSERVATION STATISTICS")
        print("="*60)
        print(f"File: {os.path.basename(self.csv_file)}")
        print(f"Duration: {self.data['timestamp'].iloc[-1] - self.data['timestamp'].iloc[0]:.2f} seconds")
        print(f"Total Points: {len(self.data)}")
        print(f"EEF Position Ranges:")
        for axis in ['x', 'y', 'z']:
            col = f'eef_pos_{axis}'
            print(f"  {axis.upper()}: [{self.data[col].min():.4f}, {self.data[col].max():.4f}] m")
        print("="*60)

    def plot_all(self, show_stats=True, save_html=None, show_plots=True):
        """Plot all visualizations"""
        if not self.load_data():
            return False
        if show_stats:
            self.print_statistics()
        print("\n📊 Creating plots...")
        traj_fig = self.create_3d_eef_trajectory_plot(smooth_window=5)
        pos_fig = self.create_position_time_plot(smooth_window=5)
        quat_fig = self.create_quaternion_time_plot(smooth_window=5)
        grip_fig = self.create_gripper_plot(smooth_window=5)
        cube_fig = self.create_cube_positions_plot(smooth_window=5)
        dashboard_fig = self.create_combined_dashboard(smooth_window=5)
        if save_html:
            print(f"💾 Saving plots to {save_html}...")
            traj_fig.write_html(save_html.replace('.html', '_trajectory.html'))
            pos_fig.write_html(save_html.replace('.html', '_position.html'))
            quat_fig.write_html(save_html.replace('.html', '_quaternion.html'))
            grip_fig.write_html(save_html.replace('.html', '_gripper.html'))
            cube_fig.write_html(save_html.replace('.html', '_cubes.html'))
            dashboard_fig.write_html(save_html.replace('.html', '_dashboard.html'))
        if show_plots:
            traj_fig.show()
            pos_fig.show()
            quat_fig.show()
            grip_fig.show()
            cube_fig.show()
            dashboard_fig.show()
        return True

def main():
    parser = argparse.ArgumentParser(description="Plot Observations CSV Data")
    parser.add_argument("--csv", type=str, required=True,
                       help="Path to observations CSV file")
    parser.add_argument("--trial", type=int, default=None,
                       help="Trial to plot (if not provided, plots all data)")
    parser.add_argument("--no-stats", action="store_true",
                       help="Don't print statistics")
    parser.add_argument("--save-html", type=str, default=None,
                       help="Save plots as HTML files (provide base filename)")
    parser.add_argument("--no-show", action="store_true",
                       help="Don't show interactive plots")
    args = parser.parse_args()
    plotter = ObservationsPlotter(args.csv, trial_id=args.trial)
    plotter.plot_all(
        show_stats=not args.no_stats,
        save_html=args.save_html,
        show_plots=not args.no_show
    )

if __name__ == "__main__":
    main()