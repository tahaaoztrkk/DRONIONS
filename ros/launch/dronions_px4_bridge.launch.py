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
    # Downward lidar. The forward fan cannot see what the drone is above, and
    # that gap crashed flights: after climbing over the 3 m wall the drone
    # descended toward a target it had just spotted and flew into the top of
    # the wall it was still standing over.
    lidar_down = f'{base}/lidar_down_link/sensor/lidar_down/scan'
    # Segmentation, for building training labels. Bridged alongside rather than
    # in a separate launch because a mask taken from a different run is a mask
    # of a different frame, and the whole value of it is that mask pixel and
    # image pixel are the same pixel. Flight never subscribes, so the cost is
    # one idle bridge.
    # Flat, not the long sensor path the others use: the sensor declares
    # <topic>segmentation</topic>, and that is absolute.
    seg = '/segmentation/labels_map'

    return [Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            f'{cam}@sensor_msgs/msg/Image@gz.msgs.Image',
            f'{cam_info}@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo',
            f'{lidar}@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan',
            f'{lidar_down}@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan',
            f'{seg}@sensor_msgs/msg/Image@gz.msgs.Image',
            '--ros-args',
            '-r', f'{cam}:=/camera/image_raw',
            '-r', f'{cam_info}:=/camera/camera_info',
            '-r', f'{lidar}:=/scan',
            '-r', f'{lidar_down}:=/scan_down',
            '-r', f'{seg}:=/camera/segmentation',
        ],
        output='screen',
    )]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('world', default_value='dronions_scenario'),
        DeclareLaunchArgument('model', default_value='x500_dronions_0'),
        OpaqueFunction(function=_bridge),
    ])
