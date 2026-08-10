"""
Launches the DRONIONS Gazebo simulation world (vehicle_blue with a front
camera + a findable cardboard box) and bridges its Gazebo Transport topics
to ROS 2 topics that dronions_ros_node.py consumes.

Usage:
    ros2 launch ros/launch/dronions_sim.launch.py
"""
import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch_ros.actions import Node

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
WORLD_PATH = os.path.join(THIS_DIR, '..', 'worlds', 'dronions_diff_drive_camera.sdf')


def generate_launch_description():
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')),
        launch_arguments={
            'gz_args': f'-r {WORLD_PATH}'
        }.items(),
    )

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/model/vehicle_blue/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
            '/model/vehicle_blue/odometry@nav_msgs/msg/Odometry@gz.msgs.Odometry',
            '/camera@sensor_msgs/msg/Image@gz.msgs.Image',
            '/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo',
            '/lidar@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan',
            '--ros-args',
            '-r', '/model/vehicle_blue/cmd_vel:=/cmd_vel',
            '-r', '/model/vehicle_blue/odometry:=/odom',
            '-r', '/camera:=/camera/image_raw',
            '-r', '/camera_info:=/camera/camera_info',
            '-r', '/lidar:=/scan',
        ],
        parameters=[{
            'qos_overrides./model/vehicle_blue.subscriber.reliability': 'reliable',
        }],
        output='screen',
    )

    return LaunchDescription([
        gz_sim,
        bridge,
    ])
