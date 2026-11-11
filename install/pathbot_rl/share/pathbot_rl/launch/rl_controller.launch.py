from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('model', default_value=''),
        DeclareLaunchArgument('loop', default_value='false',
                              description='true: follow the waypoint loop; false: RViz goals'),
        Node(
            package='pathbot_rl',
            executable='rl_controller.py',
            parameters=[{'model': LaunchConfiguration('model'),
                         'loop': LaunchConfiguration('loop')}],
            output='screen',
        ),
    ])
