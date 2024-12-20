# This Python program demonstrates a basic frontier exploration algorithm for TurtleBot4.
# It analyzes the occupancy grid map to detect frontiers and sets goals towards these frontiers.
# Adapt and enhance this code for specific requirements and hardware configurations.

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, MapMetaData, Odometry
from geometry_msgs.msg import PoseStamped, Point, PointStamped
from action_msgs.msg import GoalStatusArray
from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Directions, TurtleBot4Navigator
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import math
import random
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from image_finder import Yolov2349865
from get_pose_pnp import GetPosePnp
import tf2_ros
import numpy as np

class FrontierExplorationNode(Node):
    def __init__(self, yolo, pnp):
        super().__init__('frontier_exploration_node')
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

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
        # self.marker_publisher = self.create_publisher(Marker, '/visualization_marker', 10)
        self.marker_array_publisher = self.create_publisher(MarkerArray, '/visualization_marker_array', 10)
        
        # Initial states
        self.map_data = None
        self.map_array = None
        self.map_metadata = None
        self.goal_reached = True
        self.robot_x = 0.0
        self.robot_y = 0.0

        self.image_subscriber = self.create_subscription(CompressedImage,
            '/oakd/rgb/preview/image_raw/compressed',
            self.callback,
            10
        )
        self.bridge = CvBridge()
        self.yolo = yolo
        self.pnp = pnp
        self.lastest_image = None
        self.timer2 = self.create_timer(0.1, self.image_callback)

        self.object_a_marker= None
        self.object_b_marker = None

    def callback(self, image):
        self.lastest_image = image

    def map_callback(self, msg):
        """Callback to process the incoming map data."""
        self.map_data = msg.data
        self.map_array = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.map_metadata = msg.info

    def image_callback(self):
        if self.lastest_image is not None:
            frame = self.bridge.compressed_imgmsg_to_cv2(self.lastest_image, desired_encoding='bgr8')
            res = self.yolo.image_resize(frame)
            if res is not None:
                image_num, pose_3D, pose_2D = res
                if image_num is not 0:
                    R, tvec = self.pnp.img_matrices(pose_3D, pose_2D)
                    print("find image")
                    coor = self.transform_camera_to_map(tvec[0], tvec[1], tvec[2])
                    self.delete_marker(image_num)
                    
                    if image_num == 1:
                        self.object_a_marker = self.publish_marker(coor.point.x, coor.point.y, image_num)
                    
                    elif image_num == 2:
                        self.object_b_marker = self.publish_marker(coor.point.x, coor.point.y, image_num)
                    
                    self.publish_markers()
                    print(f'{image_num}num coor x:{coor.point.x} y:{coor.point.y}')
            

            
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
            (x, y) for x, y in frontiers if not self.is_near_wall(x, y, threshold=1)
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

    def publish_marker(self, x,y, id):
        marker = Marker()
        marker.header.frame_id = "map"  # 마커의 프레임 설정
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "point"
        marker.id = id
        marker.type = Marker.SPHERE  # 동그란 마커로 설정
        marker.action = Marker.ADD
        marker.pose.position = Point(x=x, y=y, z=5.0)  # 위치 설정
        marker.pose.orientation.w = 1.0  # 회전 정보 (회전하지 않도록 설정)
        marker.scale.x = 10  # 구의 크기 설정
        marker.scale.y = 10
        marker.scale.z = 10
        marker.color.a = 1.0  # 불투명도
        marker.color.r = 1.0  # 색상 (빨간색)
        marker.color.g = 0.0
        marker.color.b = 0.0

        return marker

    def publish_markers(self):
        markers = MarkerArray()

        if self.object_b_marker:
            markers.markers.append(self.object_b_marker)
        
        if self.object_a_marker:
            markers.markers.append(self.object_a_marker)
        
        self.marker_array_publisher(markers)

    
    def delete_marker(self, id):
        marker_delete = Marker()
        marker_delete.header.frame_id = "map"
        marker_delete.header.stamp = self.get_clock().now().to_msg()
        marker_delete.ns = "point"
        marker_delete.id = id
        marker_delete.action = Marker.DELETE  # 삭제할 때는 DELETE로 설정
        
        self.marker_publisher.publish(marker_delete)

    def transform_camera_to_map(self, x, y, z):
        R = np.array([
            [0, 0, 1], 
            [-1, 0, 0], 
            [0, -1, 0]  
        ])

        original_coords = np.array([x, y, z])

        # 좌표 변환 수행
        transformed_coords = np.dot(R, original_coords)
        transform = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())

        point = PointStamped()
        point.header.frame_id = 'base_link'
        point.point.x = transformed_coords[0]
        point.point.y = transformed_coords[1]
        point.point.z = transformed_coords[2]

        transformed_point = self.tf_buffer.transform(point, 'map')
        

        return transformed_point


        




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
