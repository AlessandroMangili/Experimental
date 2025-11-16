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

def quaternion_to_yaw(x, y, z, w):
    """Convert quaternion to yaw (rotation around Z axis)"""
    return math.atan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z))

def normalize_angle(angle):
    """Normalize the angle between [-pi, pi]"""
    return math.atan2(math.sin(angle), math.cos(angle))

class ScanMarkers(Node):
    def __init__(self):
        super().__init__('scan_markers')
        self.markers = {}                                               # [id marker] = Robot's angle after detecting the marker
        self.yaw = None                                                 # Current robot yaw orientation
        self.expected_markers = 5                                       # Number of markes expected
        self.bridge = CvBridge()
        self.cv_image = None                                            # Used to save the image already converted into a cv image
        self.header = None
        self.vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.image_pub = self.create_publisher(Image, 'camera/image_with_circle', 10)
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom,
            100
        )
        self.markers_sub = self.create_subscription(
            ArucoDetection,
            '/aruco_detections',
            self.marker_detection,
            10
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
        #self.get_logger().info(f'Current yaw: {math.degrees(self.yaw):.2f}°')
        
    def marker_detection(self, msg):
        """
        Perform the marker detection by retrieving the ID and position. Calculate the angle of the camera between the x-axis and z-axis, 
        then determine the normalized global yaw. Compute the score as the absolute value of the camera angle. 
        Update the dictionary continuously, keeping the entry with the lowest score, which indicates that the marker is centered in the camera view
        """
        if not msg.markers or self.yaw is None:
            return
    
        for marker in msg.markers:
            marker_id = int(marker.marker_id)     
            px = marker.pose.position.x
            py = marker.pose.position.y
            pz = marker.pose.position.z
            if px == 0 and py == 0 and pz == 0:
                continue
            
            angle_cam = math.atan2(px, pz)
            camera_mounting_offset = 0.0
            global_yaw = normalize_angle(self.yaw + angle_cam + camera_mounting_offset)
            score = abs(angle_cam)  # smaller = more centered

            prev = self.markers.get(marker_id)
            # Keep the best estimation
            if prev is None or score < prev[1]:
                self.markers[marker_id] = (global_yaw, score)
                #self.get_logger().info(f'Best pose for marker {marker_id} updated: angle_cam={math.degrees(angle_cam):.1f}° score={score:.3f}')
                
    def image_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV image
            self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.header = msg.header
        except Exception as e:
            self.get_logger().error(f'cv_bridge exception: {e}')
                
    def _safe_publish(self, twist):
        """Publish twist only if context still valid"""
        try:
            if rclpy.ok():
                self.vel_pub.publish(twist)
        except Exception as e:
            self.get_logger().warning(f'Publish failed: {e}')
            
    def draw_circle(self, id):
        if self.cv_image is None:
            self.get_logger().warning(f'No image has been saved')
            return
        # Draw a red circle in the center
        center = (self.cv_image.shape[1] // 2, self.cv_image.shape[0] // 2)
        cv2.circle(self.cv_image, center, 50, (0, 0, 255), 3)

        cv2.namedWindow(f'marker-{id}', cv2.WINDOW_AUTOSIZE)  
        # Show the image
        cv2.imshow(f'marker-{id}', self.cv_image)
        cv2.waitKey(30)

        # Convert back to ROS Image and publish
        out_msg = self.bridge.cv2_to_imgmsg(self.cv_image, encoding='bgr8')
        out_msg.header = self.header  # preserve original header
        self.image_pub.publish(out_msg)
        
    def rotate_360(self, angular_speed):
        """
        First, the robot rotates 360 degrees to detect all the markers around itself. 
        Then it performs individual detections, starting from the marker with the lowest ID. For each marker, the robot rotates clockwise or counterclockwise, 
        depending on the shortest path to align with the marker, stopping when the marker is centered in the camera. Using the OpenCV library, a circle is drawn 
        around the marker. The robot then proced to the next marker, and repeats this process for all remaining markers
        """
        
        self.get_logger().info('Waiting for first odom message...')
        
        # Wait 10 seconds or until self.yaw is inizialized
        wait_count = 0
        while self.yaw is None and wait_count < 200:  # ad es. 200 * 0.05 = 10s timeout
            rclpy.spin_once(self, timeout_sec=0.05)
            wait_count += 1
            
        if self.yaw is None:
            self.get_logger().error('No odom received, aborting.')
            return
        
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = angular_speed
        self.get_logger().info(f'Rotation begin: speed={angular_speed:.3f} rad/s')
        
        last_yaw = self.yaw
        cumulative_rotation = 0.0
        try:
            # Keep turning until the markers detected are five and he performed a whole 360°
            while (len(self.markers) < self.expected_markers) or (cumulative_rotation < (2.0 * math.pi - 0.1)):
                twist.angular.z = angular_speed
                self._safe_publish(twist)

                # lascia tempo per ricevere messaggi
                rclpy.spin_once(self, timeout_sec=0.05)

                # aggiorna integrazione dell'angolo (gestisce wrapping)
                if self.yaw is not None:
                    delta = normalize_angle(self.yaw - last_yaw)
                    cumulative_rotation += abs(delta)
                    last_yaw = self.yaw
        except KeyboardInterrupt:
            self.get_logger().info('Interrupted during rotation.')
            return

        twist.angular.z = 0.0
        self._safe_publish(twist)
        self.get_logger().info('SCANNING COMPLETED!')
        
        try:
            self.destroy_subscription(self.markers_sub)
        except Exception:
            return

        time.sleep(5)
        
        try:
            threshold = 0.1     # Threshold for aligning the robot with the marker (rads)
            while len(self.markers) != 0:
                lowest_id = min(self.markers.keys())
                marker_yaw = self.markers[lowest_id][0]
                self.get_logger().info(f'Aligning to marker {lowest_id} (target yaw={math.degrees(marker_yaw):.1f}°)')
                while True:
                    rclpy.spin_once(self, timeout_sec=0.01)
                    delta = normalize_angle(marker_yaw - self.yaw)
                    if abs(delta) <= threshold:
                        self.get_logger().info(f'Marker {lowest_id} detected, yaw={math.degrees(self.yaw):.1f}°')
                        self.draw_circle(lowest_id)
                        break    
                                    
                    twist.angular.z = angular_speed if delta > 0.0 else -angular_speed
                    self._safe_publish(twist)
                    #Sself.get_logger().info(f'ROBOT {self.yaw} --------- MARKER {marker_yaw}')                    
                    
                twist.angular.z = 0.0
                self._safe_publish(twist)
                self.markers.pop(lowest_id)
                time.sleep(5)
        except KeyboardInterrupt:
            self.get_logger().info('Interrupted during alignment.')
            return
        print("TASK COMPLETED!")
        
def main(args=None):
    rclpy.init(args=args)
    node = ScanMarkers()
    try:
        node.rotate_360(angular_speed=0.25)
    finally:
        try:
            node.destroy_node()
            input("Press a key to end up")
            cv2.destroyAllWindows()
        except Exception:
            pass

        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass

if __name__ == '__main__':
    main()