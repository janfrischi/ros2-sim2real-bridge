#!/usr/bin/env python3
"""
Live plotter for policy outputs from the /policy_outputs topic.
Displays joint positions, gripper command, and position errors in real-time.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import String
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from collections import deque
import argparse


class PolicyOutputPlotter(Node):
    """Node for live plotting of policy outputs."""

    def __init__(self, buffer_size=200, update_interval=100, show_all_data=True, window_size=30):
        super().__init__('policy_output_plotter')
        
        # Store view preferences
        self.show_all_data = show_all_data
        self.window_size = window_size
        
        self.get_logger().info('Starting Policy Output Plotter')
        self.get_logger().info(f'View mode: {"All Data" if show_all_data else f"Rolling {window_size}s Window"}')
        
        # Create subscription to policy outputs
        self.subscription = self.create_subscription(
            Float64MultiArray,
            '/policy_outputs',
            self.listener_callback,
            10)
        
        # Create subscription to current joint states (observations)
        self.observation_subscription = self.create_subscription(
            Float64MultiArray,
            '/rl/observations',
            self.observation_callback,
            10)
        
        # Add subscription to monitor policy status
        self.status_subscription = self.create_subscription(
            String,
            '/policy_status',  # Create this topic in PolicyRunner
            self.status_callback,
            10)
        
        self.get_logger().info('Subscribed to /policy_outputs, /rl/observations and /policy_status topics')
        
        # Data storage - use lists instead of deques to keep all data if needed
        if show_all_data:
            # Use regular lists for storing all data
            self.times = []
            self.data_buffers = [[] for _ in range(8)]
            self.error_times = []
            self.error_buffers = [[] for _ in range(7)]  # Store position errors for 7 joints
            self.current_positions = [[] for _ in range(7)]  # Store current joint positions
        else:
            # Use deques with fixed size for rolling window
            self.buffer_size = buffer_size
            self.times = deque(maxlen=buffer_size)
            self.data_buffers = [deque(maxlen=buffer_size) for _ in range(8)]
            self.error_times = deque(maxlen=buffer_size)
            self.error_buffers = [deque(maxlen=buffer_size) for _ in range(7)]
            self.current_positions = [deque(maxlen=buffer_size) for _ in range(7)]
        
        self.joint_names = [f'Joint {i+1}' for i in range(7)] + ['Gripper']
        self.colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'orange']
        
        # Track when we received the first message
        self.start_time = None
        self.last_timestamp = 0
        
        # Track latest policy outputs and observations for error calculation
        self.latest_policy_outputs = [0.0] * 7
        self.latest_observations = [0.0] * 7
        self.latest_policy_timestamp = 0.0
        self.latest_observation_timestamp = 0.0
        
        # Track policy pause events
        self.pause_events = []  # List of (timestamp, is_paused) tuples
        self.is_policy_paused = False
        self.pause_patches = []  # To store the pause indicator rectangles
        
        # Setup the plot
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(10, 12))
        self.fig.suptitle('Policy Outputs and Position Errors Live Plot')
        
        # Initialize lines for joint positions (top plot)
        self.joint_lines = []
        for i in range(7):
            line, = self.ax1.plot([], [], label=self.joint_names[i], color=self.colors[i])
            self.joint_lines.append(line)
        
        self.ax1.set_ylabel('Joint Position')
        self.ax1.set_title('Joint Positions')
        self.ax1.legend(loc='upper right')
        self.ax1.grid(True)
        
        # Initialize line for gripper command (middle plot)
        self.gripper_line, = self.ax2.plot([], [], label=self.joint_names[7], color=self.colors[7], linewidth=2)
        
        # Add horizontal lines for gripper command thresholds
        self.ax2.axhline(y=-1.0, color='r', linestyle='--', alpha=0.7, label='Close threshold')
        self.ax2.axhline(y=1.0, color='g', linestyle='--', alpha=0.7, label='Open threshold')
        
        self.ax2.set_ylabel('Gripper Command')
        self.ax2.set_title('Gripper Command')
        self.ax2.legend(loc='upper right')
        self.ax2.grid(True)
        
        # Initialize lines for position errors (bottom plot)
        self.error_lines = []
        for i in range(7):
            line, = self.ax3.plot([], [], label=f"Error {self.joint_names[i]}", color=self.colors[i])
            self.error_lines.append(line)
        
        self.ax3.set_ylabel('Position Error (rad)')
        self.ax3.set_xlabel('Time (seconds)')
        self.ax3.set_title('Joint Position Errors')
        self.ax3.legend(loc='upper right')
        self.ax3.grid(True)
        
        # Add view mode indicator
        view_mode_text = f"View Mode: {'Complete History' if show_all_data else f'Last {window_size}s'}"
        self.view_mode_text = self.fig.text(0.5, 0.01, view_mode_text, 
                                         ha='center', transform=self.fig.transFigure)
        
        # Add keyboard shortcut hint
        self.keyboard_hint = self.fig.text(0.99, 0.01, "Press 'v' to toggle view mode", 
                                        ha='right', fontsize=8, transform=self.fig.transFigure)
        
        # Timestamp display
        self.timestamp_text = self.fig.text(0.01, 0.01, 'Time: 0.00s', transform=self.fig.transFigure)
        
        # Animation setup
        self.ani = FuncAnimation(
            self.fig, self.update_plot,
            interval=update_interval, blit=False)
        
        # Data received indicator
        self.data_received = False
        self.get_logger().info('Plot initialized. Waiting for data...')
        
        # Set up key press event handler for toggling view mode
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

    def listener_callback(self, msg):
        """Process incoming policy output messages."""
        if not self.data_received:
            self.get_logger().info('First policy output received. Starting plot...')
            self.data_received = True
            
        # Get current timestamp relative to start time
        if self.start_time is None:
            self.start_time = self.get_clock().now().nanoseconds / 1e9
            timestamp = 0.0
        else:
            current_time = self.get_clock().now().nanoseconds / 1e9
            timestamp = current_time - self.start_time
        
        # Store timestamp
        self.times.append(timestamp)
        self.last_timestamp = timestamp
        
        # Get data from message (should be 8 values: 7 joint positions + gripper command)
        data = msg.data
        if len(data) != 8:
            self.get_logger().warning(f'Unexpected data length: {len(data)}, expected 8')
            return
        
        # Store each value in its corresponding buffer
        for i, value in enumerate(data):
            self.data_buffers[i].append(value)
            
        # Store latest policy outputs for error calculation
        for i in range(7):  # Only store joint positions, not gripper command
            self.latest_policy_outputs[i] = data[i]
        self.latest_policy_timestamp = timestamp
        
        # Calculate and store position errors if we have both policy outputs and observations
        self.calculate_errors(timestamp)

    def observation_callback(self, msg):
        """Process incoming observation messages (current joint states)."""
        if self.start_time is None:
            return  # Can't process without start time initialized
            
        current_time = self.get_clock().now().nanoseconds / 1e9
        timestamp = current_time - self.start_time
        
        # Extract joint positions from observations (first 7 values)
        data = msg.data
        if len(data) < 7:
            self.get_logger().warning(f'Unexpected observation data length: {len(data)}, expected at least 7')
            return
            
        # Store current positions
        for i in range(7):
            self.latest_observations[i] = data[i]
            self.current_positions[i].append(data[i])
            
        self.latest_observation_timestamp = timestamp
        
        # Calculate and store position errors if we have both policy outputs and observations
        self.calculate_errors(timestamp)
    
    def calculate_errors(self, timestamp):
        """Calculate position errors if both policy outputs and observations are available."""
        # Skip if either timestamp is zero (not initialized)
        if self.latest_policy_timestamp == 0.0 or self.latest_observation_timestamp == 0.0:
            return
            
        # Calculate time difference between policy and observation
        time_diff = abs(self.latest_policy_timestamp - self.latest_observation_timestamp)
        
        # Only calculate errors if timestamps are close enough (within 0.1 seconds)
        if time_diff > 0.1:
            return
            
        # Store error timestamp
        self.error_times.append(timestamp)
        
        # Calculate and store position errors for each joint
        for i in range(7):
            error = self.latest_policy_outputs[i] - self.latest_observations[i]
            self.error_buffers[i].append(error)
    
    def status_callback(self, msg):
        """Process policy status messages."""
        timestamp = 0.0
        if self.start_time is None:
            self.start_time = self.get_clock().now().nanoseconds / 1e9
        else:
            current_time = self.get_clock().now().nanoseconds / 1e9
            timestamp = current_time - self.start_time
        
        # Check for pause/resume messages
        if msg.data == "paused" and not self.is_policy_paused:
            self.is_policy_paused = True
            self.pause_events.append((timestamp, True))
            self.get_logger().info(f"Policy paused at time {timestamp:.2f}s")
        elif msg.data == "resumed" and self.is_policy_paused:
            self.is_policy_paused = False
            self.pause_events.append((timestamp, False))
            self.get_logger().info(f"Policy resumed at time {timestamp:.2f}s")
            
    def update_plot(self, frame):
        """Update the plot with the latest data."""
        # Only update if we have data
        if not self.data_received or len(self.times) < 2:
            return
            
        # Convert data structures to lists for plotting (if using deques)
        if not isinstance(self.times, list):
            times_list = list(self.times)
            error_times_list = list(self.error_times) if self.error_times else []
        else:
            times_list = self.times
            error_times_list = self.error_times
        
        # Ensure all data arrays have the same length as times_list
        min_length = min(len(times_list), *[len(buffer) for buffer in self.data_buffers])
        
        if min_length < len(times_list):
            times_list = times_list[:min_length]
        
        # Calculate min error length if we have error data
        if error_times_list:
            min_error_length = min(len(error_times_list), 
                                  *[len(buffer) for buffer in self.error_buffers if buffer])
            if min_error_length < len(error_times_list):
                error_times_list = error_times_list[:min_error_length]
        else:
            min_error_length = 0
        
        # Update joint position lines
        for i in range(7):
            data_list = list(self.data_buffers[i]) if not isinstance(self.data_buffers[i], list) else self.data_buffers[i]
            # Ensure the data list has the same length as times_list
            data_list = data_list[:min_length]
            self.joint_lines[i].set_data(times_list, data_list)
            
        # Update gripper command line
        gripper_data = list(self.data_buffers[7]) if not isinstance(self.data_buffers[7], list) else self.data_buffers[7]
        # Ensure gripper data has same length as times_list
        gripper_data = gripper_data[:min_length]
        self.gripper_line.set_data(times_list, gripper_data)
        
        # Update position error lines
        for i in range(7):
            if min_error_length > 0:
                error_data = list(self.error_buffers[i]) if not isinstance(self.error_buffers[i], list) else self.error_buffers[i]
                error_data = error_data[:min_error_length]
                self.error_lines[i].set_data(error_times_list, error_data)
        
        # Update x-axis limits based on view mode
        if self.show_all_data:
            # Show all data from beginning
            x_min = 0
            x_max = max(self.last_timestamp + 2, 10)  # At least 10 seconds width
        else:
            # Show rolling window of data
            x_min = max(0, self.last_timestamp - self.window_size)
            x_max = self.last_timestamp + 2
        
        self.ax1.set_xlim(x_min, x_max)
        self.ax2.set_xlim(x_min, x_max)
        self.ax3.set_xlim(x_min, x_max)
        
        # Auto-scale y-axes
        if len(times_list) > 1:
            joint_data = []
            for i in range(7):
                if isinstance(self.data_buffers[i], list):
                    # If showing all data, only consider data within visible window
                    if not self.show_all_data:
                        visible_data = self.data_buffers[i][:min_length]
                    else:
                        visible_indices = [j for j, t in enumerate(times_list) if t >= x_min and t <= x_max]
                        visible_data = [self.data_buffers[i][j] for j in visible_indices if j < len(self.data_buffers[i])]
                    joint_data.append(visible_data)
                else:
                    joint_data.append(list(self.data_buffers[i])[:min_length])
            
            # Calculate min/max for visible data
            joint_min = min(min(data) if data else 0 for data in joint_data) - 0.1
            joint_max = max(max(data) if data else 0 for data in joint_data) + 0.1
            
            # Do the same for gripper data
            if isinstance(self.data_buffers[7], list):
                if not self.show_all_data:
                    visible_gripper_data = self.data_buffers[7][:min_length]
                else:
                    visible_indices = [j for j, t in enumerate(times_list) if t >= x_min and t <= x_max]
                    visible_gripper_data = [self.data_buffers[7][j] for j in visible_indices if j < len(self.data_buffers[7])]
            else:
                visible_gripper_data = list(self.data_buffers[7])[:min_length]
            
            gripper_min = min(visible_gripper_data) - 0.5 if visible_gripper_data else -3
            gripper_max = max(visible_gripper_data) + 0.5 if visible_gripper_data else 3
            
            # For error data
            error_data = []
            if min_error_length > 0:
                for i in range(7):
                    if isinstance(self.error_buffers[i], list):
                        if not self.show_all_data:
                            visible_error_data = self.error_buffers[i][:min_error_length]
                        else:
                            visible_indices = [j for j, t in enumerate(error_times_list) if t >= x_min and t <= x_max]
                            visible_error_data = [self.error_buffers[i][j] for j in visible_indices if j < len(self.error_buffers[i])]
                        error_data.append(visible_error_data)
                    else:
                        error_data.append(list(self.error_buffers[i])[:min_error_length])
                
                # Calculate min/max for visible error data
                if any(data for data in error_data):
                    error_min = min(min(data) if data else 0 for data in error_data) - 0.05
                    error_max = max(max(data) if data else 0 for data in error_data) + 0.05
                    self.ax3.set_ylim(max(-0.5, error_min), min(0.5, error_max))  # Limit to reasonable range
            
            self.ax1.set_ylim(joint_min, joint_max)
            self.ax2.set_ylim(min(-3, gripper_min), max(3, gripper_max))  # Ensure we show the threshold lines
        
        # Clear previous pause indicators
        for patch in self.pause_patches:
            patch.remove()
        self.pause_patches = []
        
        # Add shaded regions for pause periods
        if self.pause_events:
            pause_start = None
            for timestamp, is_paused in self.pause_events:
                if is_paused:
                    pause_start = timestamp
                elif pause_start is not None:
                    # Draw shaded region from pause_start to timestamp
                    if pause_start >= x_min and timestamp <= x_max:
                        # Add shaded region to all plots
                        rect1 = self.ax1.axvspan(pause_start, timestamp, alpha=0.2, color='red', label='Policy Paused')
                        rect2 = self.ax2.axvspan(pause_start, timestamp, alpha=0.2, color='red')
                        rect3 = self.ax3.axvspan(pause_start, timestamp, alpha=0.2, color='red')
                        self.pause_patches.extend([rect1, rect2, rect3])
                    pause_start = None
            
            # If currently paused, draw from last pause to current time
            if self.is_policy_paused and pause_start is not None:
                if pause_start >= x_min and self.last_timestamp <= x_max:
                    rect1 = self.ax1.axvspan(pause_start, self.last_timestamp, alpha=0.2, color='red', label='Policy Paused')
                    rect2 = self.ax2.axvspan(pause_start, self.last_timestamp, alpha=0.2, color='red')
                    rect3 = self.ax3.axvspan(pause_start, self.last_timestamp, alpha=0.2, color='red')
                    self.pause_patches.extend([rect1, rect2, rect3])
        
        # Update timestamp text
        self.timestamp_text.set_text(f'Time: {self.last_timestamp:.2f}s')
        
        # Return all artists that were updated
        artists = self.joint_lines + [self.gripper_line, self.timestamp_text] + self.error_lines
        return artists
    
    def on_key_press(self, event):
        """Handle keyboard events for the plot."""
        if event.key == 'v':
            # Toggle view mode
            self.show_all_data = not self.show_all_data
            view_mode_text = f"View Mode: {'Complete History' if self.show_all_data else f'Last {self.window_size}s'}"
            self.view_mode_text.set_text(view_mode_text)
            self.get_logger().info(f"View mode changed to: {view_mode_text}")
            
            # Force update
            self.fig.canvas.draw_idle()

    def start_plotting(self):
        """Start the matplotlib event loop."""
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1, hspace=0.3)
        plt.show()


def main(args=None):
    parser = argparse.ArgumentParser(description='Live plotting of policy outputs and position errors')
    parser.add_argument('--buffer', type=int, default=3000, help='Buffer size for plotting (default: 3000)')
    parser.add_argument('--interval', type=int, default=100, help='Plot update interval in milliseconds (default: 100)')
    parser.add_argument('--window', type=int, default=30, help='Window size in seconds for rolling view (default: 30)')
    parser.add_argument('--all-data', action='store_true', help='Show all data instead of rolling window')
    
    # Parse arguments
    parsed_args = parser.parse_args(args=rclpy.utilities.remove_ros_args(args=args)[1:])
    
    rclpy.init(args=args)
    
    # Create and run the node
    plotter = PolicyOutputPlotter(
        buffer_size=parsed_args.buffer,
        update_interval=parsed_args.interval,
        show_all_data=parsed_args.all_data,
        window_size=parsed_args.window
    )

    # Use a separate thread for ROS spinning
    import threading
    spin_thread = threading.Thread(target=lambda: rclpy.spin(plotter))
    spin_thread.daemon = True
    spin_thread.start()
    
    try:
        plotter.start_plotting()
    except KeyboardInterrupt:
        pass
    finally:
        plotter.get_logger().info('Shutting down...')
        plotter.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()