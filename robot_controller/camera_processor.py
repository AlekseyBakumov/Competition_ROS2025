import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
import cv2
from cv_bridge import CvBridge

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_processor')
        self.subscription_color = self.create_subscription(
            Image,
            'color/image',
            self.listener_callback,
            10)
        self.subscription_depth = self.create_subscription(
            Image,
            'depth/image',
            self.depth_callback,
            10)
        self.publisherImg_ = self.create_publisher(Image, 'processed/image', 10)
        self.publisherTrack_ = self.create_publisher(Int32, 'track_lines/diff', 10)
        self.bridge = CvBridge()
        self.depth_frame = None
        self.depth_frame_ready = False
        self.declare_parameter('color_camera_pixels_cutoff', 220)
        self.declare_parameter('depth_camera_road_mark_distance', 0.25)

    def depth_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding=msg.encoding)
        except Exception as e:
            print(f"CvBridge error: {e}")
            return

        dist = self.get_parameter('depth_camera_road_mark_distance').value
        if msg.encoding == '32FC1':
            self.depth_frame = np.isfinite(cv_image) & (cv_image > 0) & (cv_image < dist)
            self.depth_frame_ready = True
        elif msg.encoding == '16UC1':
            self.depth_frame = cv_image > 0 & cv_image < dist
            self.depth_frame_ready = True
        else:
            print(f"Unsupported depth encoding: {msg.encoding}")
            return

    def listener_callback(self, msg):
        if not self.depth_frame_ready:
            return

        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        self.process_track_lines(cv_image)


    def process_track_lines(self, image):
        msg = Int32()
        cutoff = self.get_parameter('color_camera_pixels_cutoff').value
        image = image[:, cutoff:-cutoff]
        depth_ = self.depth_frame.copy()
        depth_ = depth_[:, cutoff:-cutoff]

        masked_image = image.copy()
        masked_image[~depth_] = [0,0,0]
        
        mask_y, mask_w, diff = self.count_yellow_white_pixels(masked_image)

        msg.data = diff
        self.publisherTrack_.publish(msg)
        
        image_message = self.bridge.cv2_to_imgmsg(mask_w, encoding="passthrough")
        self.publisherImg_.publish(image_message)

    def count_yellow_white_pixels(self, image):
        """
        Count yellow and white pixels in an image
        Returns: (yellow_count, white_count, total_pixels)
        """
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        yellow_lower = np.array([200, 200, 0])
        yellow_upper = np.array([255, 255, 20])
        
        white_lower = np.array([230, 230, 230])
        white_upper = np.array([255, 255, 255])
        
        yellow_mask = cv2.inRange(rgb, yellow_lower, yellow_upper)
        white_mask = cv2.inRange(rgb, white_lower, white_upper)
        
        yellow_count = np.count_nonzero(yellow_mask)
        white_count = np.count_nonzero(white_mask)
        total_pixels = image.shape[0] * image.shape[1]
        
        return yellow_mask, white_mask, yellow_count - white_count




def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    rclpy.spin(image_subscriber)
    image_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
