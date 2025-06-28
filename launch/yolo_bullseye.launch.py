# yolo_bullseye_detector/launch/yolo_bullseye.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='yolo_bullseye_detector',
            executable='yolo_bullseye_node',
            name='yolo_bullseye_detector',
            output='screen'
        )
    ])

