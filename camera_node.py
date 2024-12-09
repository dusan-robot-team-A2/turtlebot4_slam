import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from image_finder import Yolov2349865

class image_node(Node):
    def __init__(self, yolo):
        super().__init__('image_node')
        self.image_subscriber = self.create_subscription(CompressedImage,
            '/oakd/rgb/preview/image_raw/compressed',
            self.image_callback,
            10
        )
        self.bridge = CvBridge()
        self.yolo = yolo
        self.image_subscriber

    def image_callback(self, image):
        frame = self.bridge.compressed_imgmsg_to_cv2(image, desired_encoding='bgr8')
        self.yolo.image_resize(frame)



def main(args = None):
    rclpy.init(args=args)
    yolo = Yolov2349865()
    node = image_node(yolo)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()