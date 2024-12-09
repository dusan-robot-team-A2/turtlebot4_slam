# This Python program demonstrates a basic frontier exploration algorithm for TurtleBot4.
# It analyzes the occupancy grid map to detect frontiers and sets goals towards these frontiers.
# Adapt and enhance this code for specific requirements and hardware configurations.

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, MapMetaData, Odometry
from geometry_msgs.msg import PoseStamped, Point
from action_msgs.msg import GoalStatusArray
from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Directions, TurtleBot4Navigator
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from visualization_msgs.msg import Marker
import numpy as np
import math
import random
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from image_finder import Yolov2349865
from get_pose_pnp import GetPosePnp

class FrontierExplorationNode(Node):
    def __init__(self, yolo, pnp):
        super().__init__('frontier_exploration_node')
        
        # Subscribing to the map topic
        self.map_subscriber = self.create_subscription(OccupancyGrid, 'map', self.map_callback, 10)
        
        # # Subscribing to goal status updates
        # self.goal_status_subscriber = self.create_subscription(
        #     GoalStatusArray,
        #     '/follow_path/_action/status',  # Adjust this topic for your specific TurtleBot4 configuration
        #     self.goal_status_callback,
        #     10)
        
        # Subscribing to odometry data
        self.odom_subscriber = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        # Publisher for goal positions
        # self.goal_publisher = self.create_publisher(PoseStamped, 'goal_pose', 10)
        
        # Timer to periodically evaluate and set goals
        self.timer = self.create_timer(5.0, self.timer_callback)

        self.action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.publisher = self.create_publisher(Marker, '/visualization_marker', 10)
        
        # Initial states
        self.map_data = None
        self.map_array = None
        self.map_metadata = None
        self.goal_reached = True
        self.robot_x = 0.0
        self.robot_y = 0.0

        self.image_subscriber = self.create_subscription(CompressedImage,
            '/oakd/rgb/preview/image_raw/compressed',
            self.image_callback,
            10
        )
        self.bridge = CvBridge()
        self.yolo = yolo
        self.pnp = pnp
        self.image_subscriber

    def map_callback(self, msg):
        """Callback to process the incoming map data."""
        self.map_data = msg.data
        self.map_array = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.map_metadata = msg.info

    def image_callback(self, image):
        frame = self.bridge.compressed_imgmsg_to_cv2(image, desired_encoding='bgr8')
        # res = self.yolo.image_resize(frame)
        # if res is not None:
        #     image_num, pose_3D, pose_2D = res
        #     if image_num is not 0:
        #         translation_matrix = self.pnp.img_matrices(pose_3D, pose_2D)
        #         print(translation_matrix)
            

    def odom_callback(self, msg):
        """Callback to update the robot's current position based on odometry data."""
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

    def goal_status_callback(self, msg):
        """Callback to monitor the status of the robot's current goal."""
        if msg.status_list:
            current_status = msg.status_list[-1].status
            if current_status == 3:  # Goal reached
                self.goal_reached = True
                self.get_logger().info("Goal reached.")
            elif current_status in [2, 4]:  # Active or aborted
                self.goal_reached = False
                self.get_logger().info("Goal active or aborted.")
            else:
                self.goal_reached = True  # Default behavior

    def timer_callback(self):
        """Timer callback to evaluate and set a new goal if necessary."""
        if self.map_array is not None:
            if self.goal_reached:
                self.get_logger().info("Goal reached. Detecting new frontiers.")
                frontiers = self.detect_frontiers()
                if frontiers:
                    # self.get_logger().info(frontiers)
                    goal = self.select_goal(frontiers)
                    if goal:
                        self.send_goal(goal[0], goal[1])
                        self.get_logger().info(str(goal))
            else:
                self.get_logger().info("Waiting to reach current goal.")

    def is_valid_cell(self, x, y):
        """Check if the cell is within the valid map range."""
        return 0 <= x < self.map_metadata.width and 0 <= y < self.map_metadata.height

    def detect_frontiers(self):
        """Detect frontiers in the map (edges between known and unknown regions)."""
        frontiers = []
        for y in range(1, self.map_metadata.height - 1):
            for x in range(1, self.map_metadata.width - 1):
                if self.map_array[y, x] == -1:  # Unknown region
                    neighbors = self.map_array[y-1:y+2, x-1:x+2].flatten()
                    if 0 in neighbors:  # Adjacent to free space
                        frontiers.append((x, y))
        return frontiers

    def select_goal(self, frontiers):
        """Select a random valid frontier as the new goal."""
        valid_frontiers = [
            (x, y) for x, y in frontiers if not self.is_near_wall(x, y, threshold=1.5)
        ]
        if valid_frontiers:
            chosen_frontier = random.choice(valid_frontiers)
            goal_x = chosen_frontier[0] * self.map_metadata.resolution + self.map_metadata.origin.position.x
            goal_y = chosen_frontier[1] * self.map_metadata.resolution + self.map_metadata.origin.position.y
            return goal_x, goal_y
        return None

    def is_near_wall(self, x, y, threshold):
        """Check if a cell is near a wall."""
        for dx in range(-threshold, threshold + 1):
            for dy in range(-threshold, threshold + 1):
                nx, ny = x + dx, y + dy
                if self.is_valid_cell(nx, ny) and self.map_array[ny, nx] == 100:  # Wall
                    return True
        return False

    def publish_goal(self, goal):
        """Publish the goal as a PoseStamped message."""
        goal_msg = PoseStamped()
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.header.frame_id = "map"
        goal_msg.pose.position = Point(x=goal[0], y=goal[1], z=0.0)
        self.goal_publisher.publish(goal_msg)
        self.get_logger().info(f"New goal set at: x={goal[0]}, y={goal[1]}")

    def send_goal(self, x, y, z=0.0, w=1.0):
        if not self.action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('NavigateToPose action server not available!')
            return

        self.goal_reached = False

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        # Set the target coordinates
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.orientation.z = z
        goal_msg.pose.pose.orientation.w = w

        self.get_logger().info(f'Sending goal: x={x}, y={y}')
        send_goal_future = self.action_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal was rejected by the server.')
            self.goal_reached = True
            return

        self.get_logger().info('Goal accepted by the server.')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def result_callback(self, future):
        result = future.result()
        if result.status == 0:  # SUCCEEDED
            self.get_logger().info('Goal succeeded!')
        else:
            self.get_logger().error('Goal failed or was canceled.')
        
        self.goal_reached = True

    def publish_frontier_marker(self,frontiers):
        id = 0
        for frontier in frontiers:
            marker = Marker()
            marker.header.frame_id = "map"  # 기준 프레임 (맵 상에서 표시)
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "circle_marker"
            marker.id = id
            id += 1
            marker.type = Marker.SPHERE  # 동그라미 마커 타입
            marker.action = Marker.ADD

            # 동그라미 위치 설정
            marker.pose.position.x = frontier[0]
            marker.pose.position.y = frontier[1]  
            marker.pose.position.z = 0.0  

            # 동그라미 크기 (반지름)
            marker.scale.x = 1.0  # x 방향 크기
            marker.scale.y = 1.0  # y 방향 크기
            marker.scale.z = 0.01  # z 방향 크기 (얇게 만들어 평면에 보이게 설정)

            # 색상 설정
            marker.color.r = 0.0  # 빨간색 비율
            marker.color.g = 1.0  # 초록색 비율
            marker.color.b = 0.0  # 파란색 비율
            marker.color.a = 1.0  # 투명도 (1.0은 불투명)

            self.publisher.publish(marker) 

def main(args=None):
    rclpy.init(args=args)
    pnp = GetPosePnp()
    yolo = Yolov2349865()
    node = FrontierExplorationNode(yolo, pnp)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
