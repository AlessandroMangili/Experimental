from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, RegisterEventHandler, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource, AnyLaunchDescriptionSource
from launch_ros.actions import Node
from launch.event_handlers import OnProcessStart
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Percorso al launch file spawn_robot
    spawn_robot_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('bme_gazebo_sensors'),
                'launch',
                'spawn_robot.launch.py'
            )
        )
    )

    # Percorso al launch file aruco_tracker
    aruco_tracker_launch = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('aruco_opencv'),
                'launch',
                'aruco_tracker.launch.xml'
            )
        )
    )

    # Nodo start_scanning, verrà lanciato dopo che i due launch precedenti sono avviati
    start_scanning_node = Node(
        package='assignment1',
        executable='start_scanning',
        name='start_scanning',
        output='screen'
    )

    # Sequenza: spawn_robot → aruco_tracker → start_scanning
    # Primo evento: quando spawn_robot parte, lancio aruco_tracker
    aruco_tracker_handler = RegisterEventHandler(
        OnProcessStart(
            target_action=spawn_robot_launch,
            on_start=[aruco_tracker_launch]
        )
    )

    # Secondo evento: quando aruco_tracker parte, lancio start_scanning
    start_scanning_handler = RegisterEventHandler(
        OnProcessStart(
            target_action=aruco_tracker_launch,
            on_start=[
                # Piccola attesa per sicurezza (ad esempio 2 secondi)
                TimerAction(period=2.0, actions=[start_scanning_node])
            ]
        )
    )

    return LaunchDescription([
        spawn_robot_launch,
        aruco_tracker_handler,
        start_scanning_handler
    ])