from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Get the package directory
    pkg_dir = get_package_share_directory('franka_rl_bridge')
    
    # Define launch arguments
    policy_path_arg = DeclareLaunchArgument(
        'policy_path',
        default_value=os.path.join(pkg_dir, 'models', 'franka_lift.pt'),
        description='Path to the policy file'
    )
    
    device_arg = DeclareLaunchArgument(
        'device',
        default_value='cpu',
        description='Device to run inference on (cpu or cuda)'
    )
    
    debug_level_arg = DeclareLaunchArgument(
        'debug_level',
        default_value='1',
        description='Debug verbosity level (0-2)'
    )
    
    # Define nodes
    joint_state_listener_node = Node(
        package='franka_rl_bridge',
        executable='joint_state_listener',
        name='joint_state_listener',
        output='screen',
        parameters=[{
            'debug_level': LaunchConfiguration('debug_level')
        }]
    )
    
    policy_runner_node = Node(
        package='franka_rl_bridge',
        executable='policy_runner',
        name='policy_runner',
        output='screen',
        parameters=[{
            'debug_level': LaunchConfiguration('debug_level')
        }],
        arguments=['--policy', LaunchConfiguration('policy_path'),
                   '--device', LaunchConfiguration('device')]
    )
    
    return LaunchDescription([
        policy_path_arg,
        device_arg,
        debug_level_arg,
        joint_state_listener_node,
        policy_runner_node
    ])