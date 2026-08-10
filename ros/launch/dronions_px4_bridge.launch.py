"""
Bridges the PX4 SITL quadrotor's Gazebo camera + lidar into the ROS 2 topics
dronions_ros_node_px4.py consumes (/camera/image_raw and /scan).

Unlike the ground-rig launch file this does NOT start Gazebo or bridge any
actuation topic: PX4 spawns its own Gazebo world, and MAVROS/MAVLink owns
actuation. This is sensors-only, and is the 4th process alongside
PX4 SITL, MAVROS and the node itself.

Topic names were read off a running instance with `gz topic -l`, not guessed.
The `_0` suffix on the model name is PX4's instance number -- override it with
the model:= argument if you ever run a second vehicle.

Usage:
    ros2 launch ros/launch/dronions_px4_bridge.launch.py
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node


def _bridge(context, *args, **kwargs):
    world = LaunchConfiguration('world').perform(context)
    model = LaunchConfiguration('model').perform(context)

    base = f'/world/{world}/model/{model}/link'
    cam = f'{base}/camera_link/sensor/camera/image'
    cam_info = f'{base}/camera_link/sensor/camera/camera_info'
    lidar = f'{base}/lidar_sensor_link/sensor/lidar/scan'

    return [Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            f'{cam}@sensor_msgs/msg/Image@gz.msgs.Image',
            f'{cam_info}@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo',
            f'{lidar}@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan',
            '--ros-args',
            '-r', f'{cam}:=/camera/image_raw',
            '-r', f'{cam_info}:=/camera/camera_info',
            '-r', f'{lidar}:=/scan',
        ],
        output='screen',
    )]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('world', default_value='dronions_scenario'),
        DeclareLaunchArgument('model', default_value='x500_dronions_0'),
        OpaqueFunction(function=_bridge),
    ])
