#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import numpy as np
import cv2
import imutils

from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist

VERBOSE = False


class ImageFeature(Node):
    def __init__(self):
        super().__init__('image_feature')

        # Publishers
        self.image_pub = self.create_publisher(
            CompressedImage,
            '/output/image_raw/compressed',
            1
        )
        self.vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            1
        )

        # Subscriber
        self.subscriber = self.create_subscription(
            CompressedImage,
            '/camera/image/compressed',
            self.callback,
            1
        )

        cv2.namedWindow('window', cv2.WINDOW_AUTOSIZE)
        self.get_logger().info('ImageFeature node started. Subscribed to /camera/image/compressed')

    def callback(self, ros_data: CompressedImage):
        """Callback for incoming compressed images"""
        if VERBOSE:
            self.get_logger().info(f'Received image of type: {ros_data.format}')

        # Convert ROS2 CompressedImage to OpenCV
        np_arr = np.frombuffer(ros_data.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Detect green object
        greenLower = (50, 50, 20)
        greenUpper = (70, 255, 255)

        blurred = cv2.GaussianBlur(image_np, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, greenLower, greenUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        center = None

        twist = Twist()

        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            if M["m00"] > 0:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                if radius > 10:
                    # Draw circle and centroid
                    cv2.circle(image_np, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                    cv2.circle(image_np, center, 5, (0, 0, 255), -1)

                    # Simple proportional control
                    twist.angular.z = -0.002 * (center[0] - 320)
                    self.vel_pub.publish(twist)
            else:
                # Default rotation if no valid detection
                twist.angular.z = 0.5
                self.vel_pub.publish(twist)
        else:
            # Rotate if nothing detected
            twist.angular.z = 0.5
            self.vel_pub.publish(twist)

        # Show processed image
        cv2.imshow('window', image_np)
        cv2.waitKey(1)

        # Publish processed image as compressed
        compressed_msg = CompressedImage()
        compressed_msg.header = ros_data.header
        compressed_msg.format = "jpeg"
        compressed_msg.data = np.array(cv2.imencode('.jpg', image_np)[1]).tobytes()
        self.image_pub.publish(compressed_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ImageFeature()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down ImageFeature node")
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
