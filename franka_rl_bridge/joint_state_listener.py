#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from franka_msgs.msg import FrankaRobotState  # Import FrankaRobotState
import numpy as np

"""
This node listens to the /joint_states topic and the gripper state topic, then publishes 
the joint positions and velocities as a flat array to the /rl/observations topic.
The flat array serves as the observation for the reinforcement learning agent.
The first 7 entries are joint positions, the next 2 entries are gripper positions,
followed by 7 joint velocities and 2 gripper velocities, and finally the end-effector position (x, y, z).
"""
class JointStateListener(Node):
    def __init__(self):
        super().__init__('joint_state_listener')
        
        # Message counter for logging frequency control
        self.msg_counter = 0
        self.log_frequency = 100  # Log every N messages

        # Initialize joint and gripper state storage
        self.joint_positions = []
        self.joint_velocities = []
        self.gripper_positions = [0.0, 0.0]  # Default to closed position
        self.gripper_velocities = [0.0, 0.0]  # Default to no velocity
        self.end_effector_position = [0.0, 0.0, 0.0]  # Initialize EE position
        self.joint_names = []
        self.gripper_names = []
        self.joint_state_received = False
        self.gripper_state_received = False
        self.franka_state_received = False  # Flag for Franka state

        # Add these variables to track previous positions and times -> Used for velocity calculation of gripper
        self.previous_gripper_positions = [0.0, 0.0]
        self.previous_gripper_time = self.get_clock().now()

        # Create subscription to joint states
        self.joint_subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10 
        )

        # Create subscription to gripper state
        self.gripper_subscription = self.create_subscription(
            JointState,
            '/fr3_gripper/joint_states',
            self.gripper_state_callback,
            10
        )

        # Create subscription to Franka robot state
        self.franka_state_sub = self.create_subscription(
            FrankaRobotState,
            '/franka_robot_state_broadcaster/robot_state',
            self.franka_state_callback,
            10
        )

        # Publish observations to /rl/observations topic
        self.observation_publisher = self.create_publisher(
            Float64MultiArray,
            '/rl/observations',
            10
        )
        
        self.get_logger().info('Joint State Listener initialized with gripper and end-effector support and detailed debug output enabled')

    def joint_state_callback(self, msg):
        # Extract joint data
        self.joint_positions = list(msg.position[:7])  # Only take the first 7 joints
        self.joint_velocities = list(msg.velocity[:7])  # Only take the first 7 joints
        self.joint_names = list(msg.name)
        self.joint_state_received = True

        # Publish observation
        self.publish_observation()

    def gripper_state_callback(self, msg):
        # Extract gripper data - typically has 2 joints for the gripper fingers
        current_time = self.get_clock().now()
        current_positions = list(msg.position)
        
        # Calculate time delta in seconds
        dt = (current_time - self.previous_gripper_time).nanoseconds / 1e9
        
        # Avoid division by zero or very small time deltas
        if dt > 0.001:  # Only calculate velocity if enough time has passed (1ms)
            # Calculate velocities as position difference over time
            self.gripper_velocities = [
                (current_positions[i] - self.previous_gripper_positions[i]) / dt 
                for i in range(len(current_positions))
            ]
            
            # Store current values for next calculation
            self.previous_gripper_positions = current_positions
            self.previous_gripper_time = current_time
        
        # Update positions and other data
        self.gripper_positions = current_positions
        self.gripper_names = list(msg.name)
        self.gripper_state_received = True

        # Publish observation
        self.publish_observation()

    def franka_state_callback(self, msg):
        """Receive Franka robot state including end-effector pose."""
        # Extract O_T_EE (End-Effector Pose in Base Frame)
        ee_pose = msg.o_t_ee  # This is a PoseStamped message
        position = ee_pose.pose.position
        self.end_effector_position = [position.x, position.y, position.z]
        self.franka_state_received = True

        # Publish observation
        self.publish_observation()

    # Callback to publish the observation
    def publish_observation(self):
        # Only publish if we've received both joint data and Franka state data
        if self.joint_state_received and self.franka_state_received:
            # Combine position and velocity into one flat array:
            # [joint_pos(7), gripper_pos(2), joint_vel(7), gripper_vel(2), ee_pos(3)] -> Total 21
            obs_vector = (self.joint_positions + self.gripper_positions +
                          self.joint_velocities + self.gripper_velocities +
                          self.end_effector_position)
            
            # Create and publish message
            obs_msg = Float64MultiArray()
            obs_msg.data = obs_vector
            self.observation_publisher.publish(obs_msg)
            
            # Increment message counter
            self.msg_counter += 1
            
            # Debug output every N messages
            if self.msg_counter % self.log_frequency == 0:
                # Topic name is /rl/observations
                topic_name = self.observation_publisher.topic_name
                
                # Detailed debugging output
                self.get_logger().info('------- OBSERVATION DETAILS -------')

                # Print topic name
                self.get_logger().info(f'Topic: {topic_name}')
                
                # Print joint positions with names
                self.get_logger().info('Joint Positions:')
                for i, (name, pos) in enumerate(zip(self.joint_names, self.joint_positions)):
                    self.get_logger().info(f'  [{i}] {name}: {pos:.6f}')
                
                # Print gripper positions with names
                self.get_logger().info('Gripper Positions:')
                for i, (name, pos) in enumerate(zip(self.gripper_names, self.gripper_positions)):
                    self.get_logger().info(f'  [{i}] {name}: {pos:.6f}')
                
                # Print joint velocities with names
                self.get_logger().info('Joint Velocities:')
                for i, (name, vel) in enumerate(zip(self.joint_names, self.joint_velocities)):
                    self.get_logger().info(f'  [{i}] {name}: {vel:.6f}')
                
                # Print gripper velocities with names
                self.get_logger().info('Gripper Velocities:')
                for i, (name, vel) in enumerate(zip(self.gripper_names, self.gripper_velocities)):
                    self.get_logger().info(f'  [{i}] {name}: {vel:.6f}')
                
                # Print EE position
                self.get_logger().info('End-Effector Position (x, y, z):')
                self.get_logger().info(f'  {self.end_effector_position[0]:.6f}, {self.end_effector_position[1]:.6f}, {self.end_effector_position[2]:.6f}')
                
                self.get_logger().info('---------------------------------')


def main(args=None):
    rclpy.init(args=args)
    node = JointStateListener()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
