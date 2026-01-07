from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction, LogInfo, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource, AnyLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    spawn_robot_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('bme_gazebo_sensors'),
                'launch',
                'spawn_robot.launch.py'
            )
        )
    )

    aruco_tracker_launch = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('aruco_opencv'),
                'launch',
                'aruco_tracker.launch.xml'
            )
        )
    )

    start_scanning_terminal = ExecuteProcess(
        cmd=[
            'gnome-terminal', '--',
            'ros2', 'run', 'assignment1', 'start_scanning_exam'
        ],
        output='screen'
    )

    sequence = [
        spawn_robot_launch,
        LogInfo(msg='spawn_robot launched, scheduling aruco_tracker in 2s'),
        TimerAction(
            period=2.0,
            actions=[
                aruco_tracker_launch,
                LogInfo(msg='aruco_tracker launched, scheduling start_scanning in new terminal in 2s'),
                TimerAction(
                    period=2.0,
                    actions=[start_scanning_terminal]
                )
            ]
        )
    ]

    return LaunchDescription(sequence)
