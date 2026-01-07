import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from aruco_opencv_msgs.msg import ArucoDetection
import math
import time
from cv_bridge import CvBridge
import cv2
from sensor_msgs.msg import Image
import numpy as np
import time

def is_centered_pixel(center, img_shape, pixel_thresh=9):
    """Check if the marker's center is centered in the camera view"""
    _, w = img_shape[:2]
    cx_img = w/2.0
    dx = center[0] - cx_img
    return abs(dx) <= pixel_thresh

class ScanMarkers(Node):
    def __init__(self):
        super().__init__('scan_markers')
        self.yaw = None                                                 # Current robot yaw orientation
        self.bridge = CvBridge()
        self.cv_image = None                                            # Used to save the image already converted into a cv image
        self.spheres = {}
        self.header = None
        self.finding = True
        self.twist = Twist()
        self.color_id_map = {
            "magenta": 10,
            "cyan": 6,
            "yellow": 2,
            "green": 1,
            "red": 8
        }
        self.color_ranges = {
            "magenta": (
                (140, 100, 100),
                (170, 255, 255)
            ),
            "cyan": (
                (85, 100, 100),
                (100, 255, 255)
            ),
            "yellow": (
                (20, 100, 100),
                (35, 255, 255)
            ),
            "green": (
                (45, 100, 100),
                (75, 255, 255)
            ),
            "red": (
                (0, 120, 70),
                (10, 255, 255)
            )
        }     
        self.vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.image_pub = self.create_publisher(Image, 'camera/image_with_circle', 10)
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom,
            100
        )
        
        self.image_sub = self.create_subscription(                      # Subscriber to the original camera topic
            Image, 
            'camera/image', 
            self.image_callback, 
            10
        )

    def odom(self, msg):
        """Perform the robot yaw orientation"""
        q = msg.pose.pose.orientation
        self.yaw = quaternion_to_yaw(q.x, q.y, q.z, q.w)
                
    def image_callback(self, msg):
        """Convert and save the picture into an opencv image"""
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.header = msg.header

            blurred = cv2.GaussianBlur(self.cv_image, (11, 11), 0)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

            # Convert ROS Image to OpenCV image
            for color_name, (lower, upper) in self.color_ranges.items():
                if color_name in self.spheres and self.finding:
                    continue

                # Create mask and clean it
                mask = cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
                mask = cv2.erode(mask, None, iterations=2)
                mask = cv2.dilate(mask, None, iterations=2)

                # Find contours robustly
                cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = cnts[0]

                if len(cnts) > 0:
                    if self.finding:
                        sphere_id = self.color_id_map[color_name]
                        self.spheres[color_name] = {
                            'id': sphere_id
                        }
                        self.get_logger().info(f'Added color: {color_name} with id: {sphere_id}')
                    else:
                        if color_name == next(iter(self.spheres)):
                            c = max(cnts, key=cv2.contourArea)
                            if cv2.contourArea(c) < 300:
                                continue

                            M = cv2.moments(c)
                            if M["m00"] == 0:
                                continue

                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])

                            error = cx - self.cv_image.shape[1] // 2
                            ((x, y), radius) = cv2.minEnclosingCircle(c)
                            if abs(error) < 10:
                                cv2.circle(self.cv_image, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                                cv2.circle(self.cv_image, (cx, cy), 4, (0, 0, 255), -1)
                                cv2.putText(
                                    self.cv_image, color_name,
                                    (cx - 20, cy - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (255, 255, 255), 2
                                )
                                out_msg = self.bridge.cv2_to_imgmsg(self.cv_image, encoding='bgr8')
                                out_msg.header = self.header
                                self.image_pub.publish(out_msg)
                                del self.spheres[color_name]
                                self.twist.angular.z = 0.0
                                self.safe_publish(self.vel_pub, self.twist)
                                time.sleep(1) 
                                self.twist.angular.z = 0.5
                                self.safe_publish(self.vel_pub, self.twist)


        except Exception as e:
            self.get_logger().error(f'Exception raised by the image conversion: {e}')
                
    def safe_publish(self, publisher, msg):
        """Publish twist only if context still valid"""
        try:
            if rclpy.ok():
                publisher.publish(msg)
        except Exception as e:
            self.get_logger().warning(f'Exception raised while publishing: {e}')        
            
    def rotate_360(self, angular_speed):        
        self.get_logger().info('Waiting for first odom message...')
        
        # Wait 10 seconds or until self.yaw is inizialized
        wait_count = 0
        while self.yaw is None and wait_count < 200:
            rclpy.spin_once(self, timeout_sec=0.05)
            wait_count += 1
            
        if self.yaw is None:
            self.get_logger().error('No odom received, aborting.')
            return
        
        self.twist.linear.x = 0.0
        self.twist.angular.z = angular_speed
        self.get_logger().info(f'Rotation begin: speed={angular_speed:.3f} rad/s')
        
        last_yaw = self.yaw
        cumulative_rotation = 0.0
        self.twist.angular.z = angular_speed

        try:
            # Keep turning until the markers detected are five and he performed a whole 360°
            while (cumulative_rotation < (2.0 * math.pi)):
                self.safe_publish(self.vel_pub, self.twist)
                rclpy.spin_once(self, timeout_sec=0.05)

                if self.yaw is not None:
                    delta = normalize_angle(self.yaw - last_yaw)
                    cumulative_rotation += abs(delta)
                    last_yaw = self.yaw
        except KeyboardInterrupt:
            self.get_logger().info('Interrupted during rotation.')
            return

        self.twist.angular.z = 0.0
        self.safe_publish(self.vel_pub, self.twist)
        self.get_logger().info('SCANNING COMPLETED!')
        self.finding = False
        self.spheres = dict(sorted(self.spheres.items(), key=lambda item: item[1]["id"]))
    
        try:
            self.twist.angular.z = 0.5
            # Keep turning until the markers detected are five and he performed a whole 360°
            while (len(self.spheres) > 0):
                self.safe_publish(self.vel_pub, self.twist)
                rclpy.spin_once(self, timeout_sec=0.05)
                #self.get_logger().info(f'MAPPA {self.spheres}')
        except KeyboardInterrupt:
            self.get_logger().info('Interrupted during rotation.')
            return
        
        self.twist.angular.z = 0.0
        self.safe_publish(self.vel_pub, self.twist)
        self.get_logger().info("TASK COMPLETED!")
        
def main(args=None):
    rclpy.init(args=args)
    node = ScanMarkers()
    try:
        node.rotate_360(angular_speed=0.4)
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass

        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass

if __name__ == '__main__':
    main()