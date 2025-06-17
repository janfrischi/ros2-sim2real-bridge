#!/usr/bin/env python3
"""
Launch file for BC Policy Runner
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Declare arguments
    policy_path_arg = DeclareLaunchArgument(
        'policy_path',
        description='Path to the trained BC policy file (.pt)'
    )
    
    device_arg = DeclareLaunchArgument(
        'device',
        default_value='cpu',
        description='Device to run inference on (cpu or cuda)'
    )
    
    deterministic_arg = DeclareLaunchArgument(
        'deterministic',
        default_value='true',
        description='Use deterministic policy inference'
    )
    
    frequency_arg = DeclareLaunchArgument(
        'frequency',
        default_value='20.0',
        description='Control frequency in Hz'
    )
    
    # BC Policy Runner Node
    bc_policy_node = Node(
        package='franka_rl_bridge',
        executable='bc_policy_runner.py',
        name='bc_policy_runner',
        parameters=[{
            'policy_path': LaunchConfiguration('policy_path'),
            'device': LaunchConfiguration('device'),
            'deterministic': LaunchConfiguration('deterministic'),
            'frequency': LaunchConfiguration('frequency'),
        }],
        output='screen'
    )
    
    return LaunchDescription([
        policy_path_arg,
        device_arg,
        deterministic_arg,
        frequency_arg,
        bc_policy_node,
    ])