import math
from collections import deque

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor, ParameterType

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge


class FollowTraceLeftYellow(Node):
    def __init__(self):
        super().__init__("follow_trace_left_yellow")

        # ====== параметры ======
        self.DIFF_CENTERS = self.declare_parameter(
            'DIFF_CENTERS', 5,
            descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER)
        ).value

        self.LINES_H_RATIO = self.declare_parameter(
            'LINES_H_RATIO', 3.0 / 4.0,
            descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE)
        ).value

        self.ANALOG_CAP_MODE = self.declare_parameter(
            'ANALOG_CAP_MODE', True,
            descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_BOOL)
        ).value

        self.linear_speed = self.declare_parameter(
            'linear_speed', 0.25,
            descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE)
        ).value

        # PID
        self.Kp = self.declare_parameter(
            'Kp', 3.0,
            descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE)
        ).value
        self.Ki = self.declare_parameter(
            'Ki', 1.0,
            descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE)
        ).value
        self.Kd = self.declare_parameter(
            'Kd', 0.25,
            descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE)
        ).value

        # Warp (масштабируемые настройки)
        self.warp_top_y_frac = self.declare_parameter('warp_top_y_frac', 0.625).value
        self.warp_top_margin_frac = self.declare_parameter('warp_top_margin_frac', 0.078).value
        self.warp_bottom_y_frac = self.declare_parameter('warp_bottom_y_frac', 1.0).value
        self.warp_bottom_margin_frac = self.declare_parameter('warp_bottom_margin_frac', 0.0).value

        # Без переворота
        self.warp_flip_vertical = self.declare_parameter('warp_flip_vertical', False).value

        # Публикация
        self.publish_cmd = self.declare_parameter('publish_cmd', True).value

        # ====== ROS ======
        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, "/color/image", self.cb_image, 10)
        self.pub_cmd = self.create_publisher(Twist, "/cmd_vel", 10)
        self.pub_dbg = self.create_publisher(Image, "processed/image", 10)

        # ====== состояния ======
        self.prev_e = 0.0
        self.E = 0.0

        # прошлые x (в координатах полного perspective)
        self.prev_yellow_x = deque([0], maxlen=1)
        self.prev_white_x = deque([0], maxlen=1)

    # ----------------- warp -----------------
    def warp_perspective(self, bgr):
        h, w = bgr.shape[:2]

        top_y = int(h * float(self.warp_top_y_frac))
        bot_y = int(h * float(self.warp_bottom_y_frac)) - 1
        top_margin = int(w * float(self.warp_top_margin_frac))
        bot_margin = int(w * float(self.warp_bottom_margin_frac))

        top_y = max(0, min(h - 1, top_y))
        bot_y = max(0, min(h - 1, bot_y))
        if bot_y <= top_y + 10:
            bot_y = min(h - 1, top_y + 80)

        # BL, BR, TL, TR
        pts1 = np.float32([
            [0 + bot_margin, bot_y],
            [w - 1 - bot_margin, bot_y],
            [0 + top_margin, top_y],
            [w - 1 - top_margin, top_y],
        ])

        out_w = int(abs(pts1[0][0] - pts1[1][0]))
        out_h = int(abs(pts1[0][1] - pts1[2][1]))

        out_w = max(80, out_w)
        out_h = max(80, out_h)

        pts2 = np.float32([
            [0, out_h - 1],
            [out_w - 1, out_h - 1],
            [0, 0],
            [out_w - 1, 0],
        ])

        M = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(bgr, M, (out_w, out_h))

        if bool(self.warp_flip_vertical):
            dst = cv2.flip(dst, 0)

        return dst

    # ----------------- line finders (ВАЖНО: дорога "жёлтая слева, белая справа") -----------------
    def find_yellow_left(self, perspective, y_row):
        """
        Ищем ЖЁЛТУЮ линию в ЛЕВОЙ половине.
        Берём самый правый жёлтый пиксель (граница ближе к центру дороги),
        и возвращаем X в координатах full perspective.
        """
        h, w = perspective.shape[:2]
        y_row = max(0, min(h - 1, y_row))

        left = perspective[:, :w // 2, :]
        hsv = cv2.cvtColor(left, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (15, 80, 80), (40, 255, 255))
        mask = cv2.dilate(mask, np.ones((2, 2), np.uint8), iterations=2)

        xs = np.where(mask[y_row] == 255)[0]
        if xs.size > 0:
            x = int(xs[-1])              # самый правый в левой половине
            self.prev_yellow_x.append(x)  # уже в full coords (потому что левая половина начинается с 0)
            return x
        return int(sum(self.prev_yellow_x) / len(self.prev_yellow_x))

    def find_white_right(self, perspective, y_row):
        """
        Ищем БЕЛУЮ линию в ПРАВОЙ половине.
        Берём самый левый белый пиксель (граница ближе к центру дороги),
        и возвращаем X в координатах full perspective.
        """
        h, w = perspective.shape[:2]
        y_row = max(0, min(h - 1, y_row))

        right = perspective[:, w // 2:, :]
        gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        mask = cv2.compare(gray, 250, cv2.CMP_GE)

        xs = np.where(mask[y_row] == 255)[0]
        if xs.size > 0:
            x = int(xs[0]) + (w // 2)     # перевод в full coords
            self.prev_white_x.append(x)
            return x
        return int(sum(self.prev_white_x) / len(self.prev_white_x))

    # ----------------- PID -----------------
    def compute_pid(self, target_angle):
        e = math.atan2(math.sin(target_angle), math.cos(target_angle))
        w = (self.Kp * e) + (self.Ki * (self.E + e)) + (self.Kd * (e - self.prev_e))
        w = math.atan2(math.sin(w), math.cos(w))
        self.E += e
        self.prev_e = e
        return w

    # ----------------- callback -----------------
    def cb_image(self, msg: Image):
        # BGR
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception:
            cv = self.bridge.imgmsg_to_cv2(msg, desired_encoding=msg.encoding)
            if msg.encoding.lower() in ('rgb8', 'rgb16'):
                bgr = cv2.cvtColor(cv, cv2.COLOR_RGB2BGR)
            else:
                bgr = cv

        perspective = self.warp_perspective(bgr)
        ph, pw = perspective.shape[:2]
        y_row = int(ph * float(self.LINES_H_RATIO))
        y_row = max(0, min(ph - 1, y_row))

        # ====== ключевое: жёлтая слева, белая справа ======
        yellow_x = self.find_yellow_left(perspective, y_row)
        white_x = self.find_white_right(perspective, y_row)

        middle_x = int((yellow_x + white_x) // 2)

        center = (pw // 2, y_row)
        target = (middle_x, y_row)

        # direction как раньше (если вдруг будет наоборот — поменяешь одну строку)
        direction = center[0] - target[0]

        twist = Twist()
        twist.linear.x = float(self.get_parameter('linear_speed').value)
        twist.angular.z = 0.0

        if abs(direction) > int(self.DIFF_CENTERS):
            angle_to_goal = math.atan2(direction, 215.0)
            angular_v = self.compute_pid(angle_to_goal)
            twist.angular.z = float(angular_v)

            if bool(self.ANALOG_CAP_MODE):
                twist.linear.x = abs(twist.linear.x * (1.0 - abs(angular_v * 0.75)))
            else:
                twist.linear.x = abs(twist.linear.x * (1.0 - abs(angular_v)))

        if bool(self.get_parameter('publish_cmd').value):
            self.pub_cmd.publish(twist)

        # ====== DEBUG ======
        dbg = perspective.copy()

        # рисуем найденные точки линий
        cv2.circle(dbg, (int(yellow_x), y_row), 5, (0, 255, 255), -1)   # yellow
        cv2.circle(dbg, (int(white_x), y_row), 5, (255, 255, 255), -1)  # white

        # центр/цель
        cv2.circle(dbg, center, 5, (0, 255, 0), -1)
        cv2.circle(dbg, target, 5, (0, 0, 255), -1)

        # стрелка (вектор) куда “рулить”
        cv2.arrowedLine(dbg, center, target, (0, 0, 255), 3, tipLength=0.25)

        if abs(direction) <= int(self.DIFF_CENTERS):
            txt = "TURN: STRAIGHT"
        elif direction > 0:
            txt = "TURN: LEFT"
        else:
            txt = "TURN: RIGHT"

        cv2.putText(
            dbg,
            f"{txt} dir(px)={direction} wz={twist.angular.z:.2f} v={twist.linear.x:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        self.pub_dbg.publish(self.bridge.cv2_to_imgmsg(dbg, encoding='bgr8'))


def main(args=None):
    rclpy.init(args=args)
    node = FollowTraceLeftYellow()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
