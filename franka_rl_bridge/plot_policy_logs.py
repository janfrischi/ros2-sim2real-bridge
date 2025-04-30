# FILE: plot_policy_logs.py
import pandas as pd
import ast
import numpy as np
import plotly.graph_objects as go

# Load the CSV file
file_path = "/home/pdzuser/franka_ros2_ws/src/franka_rl_bridge/franka_rl_bridge/policy_logs.csv"
df = pd.read_csv(file_path)

# Convert stringified lists to actual lists
columns_to_convert = [
    "joint_positions", "joint_velocities", "object_position", "target_position", "last_action", "interpreted_actions"
]
for col in columns_to_convert:
    # Preprocess the column to fix formatting issues
    df[col] = df[col].str.replace(r"\s+", ", ", regex=True)  # Replace spaces with commas
    df[col] = df[col].str.replace(r",\s*,", ", ", regex=True)  # Remove duplicate commas
    df[col] = df[col].str.replace(r"\[,", "[", regex=True)  # Fix cases like "[,0.0]"
    df[col] = df[col].str.replace(r",\]", "]", regex=True)  # Fix cases like "[0.0,]")
    df[col] = df[col].apply(ast.literal_eval)  # Convert to actual lists

# Convert timestamp from Unix format to relative time starting at 0
df["timestamp"] = df["timestamp"] - df["timestamp"].iloc[0]

# Calculate joint accelerations by differentiating joint velocities using deltas
joint_accelerations = []
timestamps = df["timestamp"].values
for velocities in df["joint_velocities"]:
    accelerations = []
    for i in range(1, len(velocities)):
        delta_v = velocities[i] - velocities[i - 1]  # Change in velocity
        delta_t = timestamps[i] - timestamps[i - 1]  # Change in time
        accelerations.append(delta_v / delta_t if delta_t != 0 else 0.0)  # Avoid division by zero
    accelerations.insert(0, 0.0)  # Assume initial acceleration is 0
    joint_accelerations.append(accelerations)
df["joint_accelerations"] = joint_accelerations

# Function to create a plot for a specific variable group
def create_plot(df, variable_name, title):
    fig = go.Figure()
    for i in range(len(df[variable_name][0])):
        fig.add_trace(go.Scatter(
            x=df["timestamp"],
            y=df[variable_name].apply(lambda x: x[i]),
            mode="lines",
            name=f"{variable_name}[{i}]"
        ))
    # Limit x-axis to [0, min(4, max timestamp)]
    x_max = min(4, df["timestamp"].max())
    fig.update_layout(
        title=dict(text=title, x=0.5),  # Center the title
        xaxis_title="Time (seconds)",
        yaxis_title="Values",
        legend_title="Variables",
        template="plotly_white",
        xaxis=dict(range=[0, x_max])
    )
    fig.show()

# Create separate plots
create_plot(df, "joint_positions", "Joint Positions Over Time")
create_plot(df, "joint_velocities", "Joint Velocities Over Time")
create_plot(df, "joint_accelerations", "Joint Accelerations Over Time")  # New plot for joint accelerations
create_plot(df, "object_position", "Object Position Over Time")
create_plot(df, "target_position", "Target Position Over Time")
create_plot(df, "last_action", "Last Action Over Time")
create_plot(df, "interpreted_actions", "Interpered Actions Over Time")  # <-- Add this plot