import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge


class TurnSignNode(Node):
    def __init__(self):
        super().__init__('turn_sign_node')
        self.bridge = CvBridge()

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel/turn_sign', 10)

        self.color_sub = self.create_subscription(Image, '/color/image', self.color_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/depth/image', self.depth_callback, 10)
        self.pub_dbg = self.create_publisher(Image, 'processed_sign/image', 10)
        
        self.img_dep = None
        self.img_dep_ready = False

        self.declare_parameter('depth_camera_intersection_distance', 0.4)

        self.img_bgr = None
        self.timer = self.create_timer(0.1, self.loop)

        self.left_angle = 0.9
        self.right_angle = -0.8
        self.linear_speed = 0.0

        self.command_duration = 0.5  # секунд держать команду после детекта

        self._active_until = 0.0
        self._last_dir = 0  # -1 left, +1 right

        # Минимальные пороги площадей
        self.min_blue_area = 2000
        self.min_arrow_area = 400

    def color_callback(self, msg: Image):
        self.img_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def depth_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding=msg.encoding)
        except Exception as e:
            print(f"CvBridge error: {e}")
            return

        dist = self.get_parameter('depth_camera_intersection_distance').value
        if msg.encoding == '32FC1':
            self.img_dep = np.isfinite(cv_image) & (cv_image > 0) & (cv_image < dist)
            self.img_dep_ready = True
        elif msg.encoding == '16UC1':
            self.img_dep = cv_image > 0 & cv_image < dist
            self.img_dep_ready = True
        else:
            print(f"Unsupported depth encoding: {msg.encoding}")
            return

    def loop(self):
        now = self.get_clock().now().nanoseconds / 1e9

        cmd = Twist()
        direction = self.detect_turn_sign_direction(self.img_bgr)

        if direction is not None:
            self._last_dir = direction
            self._active_until = now + self.command_duration

        cmd = Twist()
        if now < self._active_until and self._last_dir != 0:
            cmd.linear.x = float(self.linear_speed)
            cmd.angular.z = float(self.left_angle if self._last_dir < 0 else self.right_angle)
        else:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

        self.cmd_pub.publish(cmd)

    def detect_turn_sign_direction(self, bgr):
        """
          -1 если стрелка влево
          +1 если стрелка вправо
        """
        if bgr is None:
            return None
        if self.img_dep_ready is False:
            return None

        bgr[~self.img_dep] = [0,0,0]
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        msg_debug = self.bridge.cv2_to_imgmsg(bgr, encoding="passthrough")
        self.pub_dbg.publish(msg_debug)

        # Ищем синий круг (фон знака)
        # Под эту картинку хорошо подходит H примерно 90..130 (OpenCV HSV: H=0..179)
        blue_mask = cv2.inRange(hsv, (90, 80, 30), (130, 255, 255))
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Берем самый большой синий объект
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < self.min_blue_area:
            return None

        # Проверка круглости чтобы не ловить случайные синие объекты
        peri = cv2.arcLength(c, True)
        if peri <= 1e-3:
            return None
        circularity = 4.0 * np.pi * area / (peri * peri)
        if circularity < 0.6:
            return None

        # Центр знака
        M = cv2.moments(c)
        if M["m00"] <= 1e-6:
            return None
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Маска области знака залитый контур
        sign_mask = np.zeros(blue_mask.shape, dtype=np.uint8)
        cv2.drawContours(sign_mask, [c], -1, 255, -1)

        #Ищем стрелку как светлую область внутри знака:
        # низкая насыщенность (S) и достаточно высокий Value (V)
        arrow_mask = cv2.inRange(hsv, (0, 0, 50), (179, 60, 255))
        arrow_mask = cv2.bitwise_and(arrow_mask, arrow_mask, mask=sign_mask)
        arrow_mask = cv2.morphologyEx(arrow_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        arrow_mask = cv2.morphologyEx(arrow_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

        a_contours, _ = cv2.findContours(arrow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not a_contours:
            return None

        a = max(a_contours, key=cv2.contourArea)
        a_area = cv2.contourArea(a)
        if a_area < self.min_arrow_area:
            return None

        pts = a.reshape(-1, 2)
        min_x = int(pts[:, 0].min())
        max_x = int(pts[:, 0].max())

        # Насколько далеко стрелка уходит от центра вправо/влево
        right_extent = max_x - cx
        left_extent = cx - min_x

        # Если больше уходит вправо — стрелка вправо иначе влево
        if right_extent > left_extent:
            return +1
        else:
            return -1


def main(args=None):
    rclpy.init(args=args)
    node = TurnSignNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
