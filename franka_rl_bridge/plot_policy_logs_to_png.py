import pandas as pd
import ast
import numpy as np
import plotly.graph_objects as go
import os
from scipy.signal import savgol_filter

# Load the CSV file
file_path = "/home/pdzuser/franka_ros2_ws/src/franka_rl_bridge/franka_rl_bridge/policy_logs.csv"
df = pd.read_csv(file_path)

# Convert stringified lists to actual lists
columns_to_convert = ["joint_positions", "joint_velocities", "object_position", "target_position", "last_action"]
for col in columns_to_convert:
    df[col] = df[col].str.replace(r"\s+", ", ", regex=True)
    df[col] = df[col].str.replace(r",\s*,", ", ", regex=True)
    df[col] = df[col].str.replace(r"\[,", "[", regex=True)
    df[col] = df[col].str.replace(r",\]", "]", regex=True)
    df[col] = df[col].apply(ast.literal_eval)

# Convert timestamp from Unix format to relative time starting at 0
df["timestamp"] = df["timestamp"] - df["timestamp"].iloc[0]

# Calculate joint accelerations by differentiating joint velocities using deltas
joint_accelerations = []
timestamps = df["timestamp"].values
for velocities in df["joint_velocities"]:
    accelerations = []
    for i in range(1, len(velocities)):
        delta_v = velocities[i] - velocities[i - 1]
        delta_t = timestamps[i] - timestamps[i - 1]
        accelerations.append(delta_v / delta_t if delta_t != 0 else 0.0)
    accelerations.insert(0, 0.0)
    joint_accelerations.append(accelerations)
df["joint_accelerations"] = joint_accelerations

# Create 'plots' directory in the package root (one level up from this file)
plots_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "plots")
os.makedirs(plots_dir, exist_ok=True)

# Function to smooth data using Savitzky-Golay filter
def smooth_series(series, window_length=11, polyorder=3):
    # window_length must be odd and <= len(series)
    if len(series) < window_length:
        window_length = len(series) if len(series) % 2 == 1 else len(series) - 1
    if window_length < 3:
        return series
    return savgol_filter(series, window_length, polyorder)

# Professional color palette
COLOR_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf"
]

# Function to create and save a plot for a specific variable group
def create_and_save_plot(df, variable_name, title, filename):
    fig = go.Figure()
    n_vars = len(df[variable_name][0])
    for i in range(n_vars):
        y_raw = df[variable_name].apply(lambda x: x[i]).values
        y_smooth = smooth_series(y_raw)
        fig.add_trace(go.Scatter(
            x=df["timestamp"],
            y=y_smooth,
            mode="lines+markers",
            name=f"{variable_name}[{i}]",
            line=dict(
                color=COLOR_PALETTE[i % len(COLOR_PALETTE)],
                width=2,  # Thinner lines
                shape="spline"
            ),
            marker=dict(size=4, symbol="circle"),
            opacity=0.95
        ))
    # Limit x-axis to [0, min(4, max timestamp)]
    x_max = min(30, df["timestamp"].max())
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, family="Arial"), x=0.5),  # Larger, centered title
        xaxis_title="Time (seconds)",
        yaxis_title="Values",
        legend_title="Variables",
        template="plotly_white",
        xaxis=dict(
            range=[0, x_max],
            showgrid=True,
            gridcolor="#e5e5e5",
            zeroline=False,
            tickfont=dict(size=16),
            title=dict(text="Time (seconds)", font=dict(size=18))
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#e5e5e5",
            zeroline=False,
            tickfont=dict(size=16),
            title=dict(text="Values", font=dict(size=18))
        ),
        legend=dict(
            font=dict(size=14),
            orientation="v",
            x=1.02,
            y=1,
            xanchor="left",
            yanchor="top",
            bordercolor="#CCCCCC",
            borderwidth=1,
            bgcolor="rgba(255,255,255,0.8)"
        ),
        margin=dict(l=70, r=180, t=70, b=60),
        plot_bgcolor="#fcfcfc",
        paper_bgcolor="#fcfcfc",
        hovermode="x unified"
    )
    # Save to the 'plots' directory as SVG
    fig.write_image(os.path.join(plots_dir, filename), width=1100, height=600, scale=2)

# Create and save plots as SVGs
create_and_save_plot(df, "joint_positions", "Joint Positions Over Time", "joint_positions.svg")
create_and_save_plot(df, "joint_velocities", "Joint Velocities Over Time", "joint_velocities.svg")
create_and_save_plot(df, "joint_accelerations", "Joint Accelerations Over Time", "joint_accelerations.svg")
create_and_save_plot(df, "object_position", "Object Position Over Time", "object_position.svg")
create_and_save_plot(df, "target_position", "Target Position Over Time", "target_position.svg")
create_and_save_plot(df, "last_action", "Last Action Over Time", "last_action.svg")
