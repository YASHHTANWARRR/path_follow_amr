from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('model', default_value=''),
        Node(
            package='pathbot_rl',
            executable='rl_controller.py',
            parameters=[{'model': LaunchConfiguration('model')}],
            output='screen',
        ),
    ])
