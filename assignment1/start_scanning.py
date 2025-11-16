import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from aruco_opencv_msgs.msg import ArucoDetection
import math
import time

def quaternion_to_yaw(x, y, z, w):
    """Convert quaternion to yaw (rotation around Z axis)."""
    return math.atan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z))

def normalize_angle(angle):
    """Normalizza l'angolo in [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))

def angle_difference(a, b):
        """Differenza normalizzata a - b (risultato in [-pi, pi])."""
        return normalize_angle(a - b)

class ScanMarkers(Node):
    def __init__(self):
        super().__init__('scan_markers')
        self.vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.markers = {} # id marker -> yaw robot position
        self.yaw = None
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom,
            100
        )
        self.markers_sub = self.create_subscription(
            ArucoDetection,
            '/aruco_detections',
            self.markers_detection,
            10
        )
        
    def odom(self, msg):
        q = msg.pose.pose.orientation
        self.yaw = quaternion_to_yaw(q.x, q.y, q.z, q.w)
        #self.get_logger().info(f'Current yaw: {math.degrees(self.yaw):.2f}°')
        
    def markers_detection(self, msg):
        if not msg.markers or self.yaw is None:
            return
    
        for marker in msg.markers:
            marker_id = int(marker.marker_id)
            # controllare se non è già presente in modo da non sovrascrivere       
            px = marker.pose.position.x
            py = marker.pose.position.y
            pz = marker.pose.position.z

            if px == 0 and py == 0 and pz == 0:
                continue
            
            angle_cam = math.atan2(px, pz)
            
            camera_mounting_offset = 0.0
            global_yaw = normalize_angle(self.yaw + angle_cam + camera_mounting_offset)
            score = abs(angle_cam)  # più piccolo = più centrato

            prev = self.markers.get(marker_id)
            # tieni la stima migliore (più centrata)
            if prev is None or score < prev[1]:
                self.markers[marker_id] = (global_yaw, score)
                #self.get_logger().info(f'Best pose for marker {marker_id} updated: angle_cam={math.degrees(angle_cam):.1f}° score={score:.3f}')
        
    def rotate_360(self, angular_speed):
        """
        Rotate approx 360 degrees using time = angle/speed.
        angular_speed: rad/s (positive -> counter-clockwise)
        """
        
        self.get_logger().info('Waiting for first odom message...')
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
        
        start_yaw = self.yaw
        try:
            while (len(self.markers) != 5 or abs(start_yaw - self.yaw) >= 0.1):
                self.vel_pub.publish(twist)
                rclpy.spin_once(self, timeout_sec=0.0)
        except KeyboardInterrupt:
            twist.angular.z = 0.0
            self.vel_pub.publish(twist)
            self.get_logger().info('Interrupted during rotation.')
            pass

        twist.angular.z = 0.0
        self.vel_pub.publish(twist)
        self.get_logger().info('SCANNING COMPLETED!')
        
        try:
            self.destroy_subscription(self.markers_sub)
        except Exception:
            pass


        time.sleep(5)        
        try:
            threshold = 0.1
            while len(self.markers) != 0:
                lowest_id = min(self.markers.keys())
                marker_yaw = self.markers[lowest_id][0]
                self.get_logger().info(f'Aligning to marker {lowest_id} (target yaw={math.degrees(marker_yaw):.1f}°)')
                
                twist.angular.z = angular_speed
                while True:
                    rclpy.spin_once(self, timeout_sec=0.01)
                    
                    delta = normalize_angle(marker_yaw - self.yaw)  # target - current
                    if abs(delta) <= threshold:
                        break
                    
                    #if (abs(self.yaw - marker_yaw) <= threshold):
                    #    break
                    
                    twist.angular.z = angular_speed if delta > 0.0 else -angular_speed
                    self.vel_pub.publish(twist)
                    self.get_logger().info(f'ROBOT {self.yaw} --------- MARKER {marker_yaw}')                    
                    
                twist.angular.z = 0.0
                self.vel_pub.publish(twist)
                self.get_logger().info(f'Marker {lowest_id} passed, yaw={math.degrees(self.yaw):.1f}°')
                self.markers.pop(lowest_id)
                time.sleep(2)
        except KeyboardInterrupt:
            twist.angular.z = 0.0
            self.vel_pub.publish(twist)
            self.get_logger().info('Interrupted during alignment.')
            pass
        print("TASK COMPLETED!")
        
def main(args=None):
    rclpy.init(args=args)
    node = ScanMarkers()
    try:
        node.rotate_360(angular_speed=0.25)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()