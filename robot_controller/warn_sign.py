import rclpy
from rclpy.node import Node

import cv2
import numpy as np

from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge


class ConstructionSignDetector(Node):
    def __init__(self):
        super().__init__('construction_sign_detector')
        self.bridge = CvBridge()

        self.sub = self.create_subscription(Image, 'color/image', self.cb_img, 10)
        self.depth_sub = self.create_subscription(Image, 'depth/image', self.depth_callback, 10)

        self.pub = self.create_publisher(String, 'sign/construction', 10)
        self.pub_dbg = self.create_publisher(Image, 'sign/construction/image', 10)

        # thresholds
        self.min_triangle_area = 1000
        self.min_red_ratio = 0.06
        self.min_yellow_ratio = 0.18
        self.min_black_ratio = 0.02

        self.dist = 0.4  # meters

        self.img_dep = None
        self.img_dep_ready = False

        self.last_bgr = None
        self.timer = self.create_timer(0.1, self.loop)

    def depth_callback(self, msg: Image):
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding=msg.encoding)
        except Exception as e:
            self.get_logger().warn(f"Depth CvBridge error: {e}")
            self.img_dep_ready = False
            self.img_dep = None
            return

        if msg.encoding == '32FC1':
            # meters
            self.img_dep = np.isfinite(depth) & (depth > 0.0) & (depth < float(self.dist))
            self.img_dep_ready = True

        elif msg.encoding == '16UC1':
            # usually millimeters
            dist_mm = int(float(self.dist) * 1000.0)
            self.img_dep = (depth > 0) & (depth < dist_mm)
            self.img_dep_ready = True

        else:
            self.get_logger().warn(f"Unsupported depth encoding: {msg.encoding}")
            self.img_dep_ready = False
            self.img_dep = None

    def cb_img(self, msg: Image):
        try:
            self.last_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f"Color CvBridge error: {e}")
            self.last_bgr = None

    def loop(self):
        # всегда публикуем, даже если чего-то нет
        out = String()
        out.data = "NONE"

        if self.last_bgr is None:
            self.pub.publish(out)
            return

        bgr = self.last_bgr.copy()

        # применяем depth маску, если она готова и совпадает по размеру
        if self.img_dep_ready and self.img_dep is not None:
            if self.img_dep.shape[:2] == bgr.shape[:2]:
                bgr[~self.img_dep] = (0, 0, 0)
            else:
                # если разные размеры — не применяем, но продолжаем работать
                self.get_logger().warn_throttle(2.0, "Depth size != color size, ignoring depth mask")

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # red mask (two ranges)
        red1 = cv2.inRange(hsv, (0, 80, 80), (10, 255, 255))
        red2 = cv2.inRange(hsv, (160, 80, 80), (179, 255, 255))
        red_mask = cv2.bitwise_or(red1, red2)

        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        dbg = bgr.copy()
        found = False

        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_triangle_area:
                continue

            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.03 * peri, True)

            if len(approx) != 3:
                continue

            x, y, w, h = cv2.boundingRect(approx)
            if w < 40 or h < 40:
                continue

            roi_hsv = hsv[y:y + h, x:x + w]
            roi_red = red_mask[y:y + h, x:x + w]

            roi_area = float(w * h)
            red_ratio = float(np.count_nonzero(roi_red)) / max(1.0, roi_area)
            if red_ratio < self.min_red_ratio:
                continue

            yellow_mask = cv2.inRange(roi_hsv, (15, 70, 70), (40, 255, 255))
            black_mask = cv2.inRange(roi_hsv, (0, 0, 0), (179, 255, 60))

            yellow_ratio = float(np.count_nonzero(yellow_mask)) / max(1.0, roi_area)
            black_ratio = float(np.count_nonzero(black_mask)) / max(1.0, roi_area)

            if yellow_ratio < self.min_yellow_ratio:
                continue
            if black_ratio < self.min_black_ratio:
                continue

            found = True
            cv2.drawContours(dbg, [approx], -1, (0, 255, 0), 3)
            cv2.putText(dbg, "CONSTRUCTION SIGN", (x, max(0, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            break

        out.data = "FOUND" if found else "NONE"
        self.pub.publish(out)

        # debug image публикуем всегда
        self.pub_dbg.publish(self.bridge.cv2_to_imgmsg(dbg, encoding='bgr8'))


def main(args=None):
    rclpy.init(args=args)
    node = ConstructionSignDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
    