from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction, LogInfo, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource, AnyLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Include del launch che spawn del robot
    spawn_robot_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('bme_gazebo_sensors'),
                'launch',
                'spawn_robot.launch.py'
            )
        )
    )

    # Include del launch aruco
    aruco_tracker_launch = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('aruco_opencv'),
                'launch',
                'aruco_tracker.launch.xml'
            )
        )
    )

    # Nodo start_scanning in un nuovo terminale
    start_scanning_terminal = ExecuteProcess(
        cmd=[
            'gnome-terminal', '--',  # apre un nuovo terminale (GNOME)
            'ros2', 'run', 'assignment1', 'start_scanning'
        ],
        output='screen'
    )

    # Sequenza usando TimerAction
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
