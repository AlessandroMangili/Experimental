# Scan aruco markers ROS2

This project allows a robot to detect ArUco markers using ROS2 and OpenCV, rotate to center them in the camera view, and visualize the detection in real-time.

## Overview

The `ScanMarkers` node performs the following steps:

1. Rotates the robot 360° to detect all markers around it.
2. Stores the robot's orientation estimate (`yaw`) when each marker is best centered in the camera view.
3. For each detected marker (starting from the lowest ID), rotates the robot until the marker is perfectly centered in the camera.
4. Draws a circle around the marker using OpenCV, displays it on the screen, and publishes the image on the ROS2 topic `/camera/image_with_circle`.
5. Repeats this process for all markers in ascending order of their ID.

### Features

* **Debug Mode**: Set `DEBUG = True` to print detailed information about robot orientation, marker detection, and centering process.
* **Expected Markers**: Set `self.EXCPECTED_MARKERS = 5` to the number of markers to detect. If there are more markers, modify this number accordingly.
* **Marker Centering**: The robot ensures a marker is centered by checking consecutive frames (default `REQUIRED_CONSECUTIVE = 3`) before considering it properly aligned.

## Prerequisites

Before running this package, ensure you have the following dependencies installed:

1. **bme_gazebo_sensors**
   GitHub: [https://github.com/your_link_here/bme_gazebo_sensors](https://github.com/CarmineD8/erl1_sensors)

2. **aruco_opencv**
   GitHub: [https://github.com/your_link_here/aruco_opencv](https://github.com/fictionlab/ros_aruco_opencv)

Make sure to download and extract these repositories, and follow their installation instructions.

## Installation

1. Clone this repository and extract it into your ROS2 workspace:

```bash
cd ~/ros2_ws/src
git clone https://github.com/AlessandroMangili/Experimental
```

2. Make sure you have the required packages (`bme_gazebo_sensors` and `aruco_opencv`) installed.

3. Compile your workspace:

```bash
cd ~/ros2_ws
colcon build
source install/setup.bash
```

## Running the Node

Launch the package using the provided launch file:

```bash
ros2 launch assignment1 start_assignment.launch.py
```

### Behavior

* At startup, the robot will rotate 360° until all markers (`self.EXCPECTED_MARKERS`) are detected.
* It then aligns to each marker starting from the lowest ID:

  * Rotates until the marker is centered in the camera.
  * Draws a circle around the marker in the OpenCV window.
  * Publishes the image with the circle on the ROS2 topic `/camera/image_with_circle`.
* The process continues until all markers have been centered and displayed.

## Debugging

To enable debug mode:

```python
DEBUG = True
```

This will print detailed information including:

* Current robot yaw in degrees
* Marker detection updates
* Centering process for each marker

## Customization

* **Number of markers**: Adjust `self.EXCPECTED_MARKERS` to match the number of markers in your environment.
* **Required consecutive frames for centering**: Adjust `self.REQUIRED_CONSECUTIVE` if needed.

## Notes

* Press **`q`** in any OpenCV window to quit the program once detection is completed.
* Make sure your robot is in a clear environment with all markers visible and around the robot to ensure proper detection.
