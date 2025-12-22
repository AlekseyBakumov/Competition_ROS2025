import time
from collections import deque
from types import SimpleNamespace

# модули
# конфигуратор
from module.config import (
    DIFF_CENTERS,
    LINES_H_RATIO,
    MAX_LINIEAR_SPEED,
    ANALOG_CAP_MODE,

    WHITE_MODE_CONSTANT,
    YELLOW_MODE_CONSTANT,
    FOLLOW_ROAD_CROP_HALF,
)


import rclpy
from rclpy.node import Node

from ament_index_python.packages import get_package_share_directory

from std_msgs.msg import Empty
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from tf_transformations import euler_from_quaternion
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

import cv2
import math
import numpy as np
from ultralytics import YOLO


class Follow_Trace_Node(Node):

    def __init__(self, linear_speed=MAX_LINIEAR_SPEED):

        super().__init__("Follow_Trace_Node")

        self.point_status = True

        self._pose_sub = self.create_subscription(Odometry, '/odom', self.pose_callback, 10)
        self._robot_Ccamera_sub = self.create_subscription(Image, "/color/image", self._callback_camera, 3)
        self._robot_cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self._sign_finish = self.create_publisher(String ,'/robot_finish', 10)
        self._lidar_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        
        self._cv_bridge = CvBridge()
        self.pose = Odometry()
        self.lidar_data = LaserScan()
        
        self._linear_speed = linear_speed
        self._prev__linear_speed = linear_speed
        self._yellow_prevs = deque([0], maxlen=1)
        self.__white_prevs = deque([0], maxlen=1)

        self.Kp = self.declare_parameter('Kp', value=3.0, descriptor=ParameterDescriptor(
            type=ParameterType.PARAMETER_DOUBLE)).get_parameter_value().double_value
        self.Ki = self.declare_parameter('Ki', value=1, descriptor=ParameterDescriptor(
            type=ParameterType.PARAMETER_DOUBLE)).get_parameter_value().double_value
        self.Kd = self.declare_parameter('Kd', value=0.25, descriptor=ParameterDescriptor(
            type=ParameterType.PARAMETER_DOUBLE)).get_parameter_value().double_value

        self.avoidance = 0
        self.prev_e = 0
        self.E = 0
        self.parking = SimpleNamespace(state=0, side=0)
        self.finish_parking = 0
        self.angle = 0

        self.STATUS_CAR = 0
        self.TASK_LEVEL = 0

        self.MAIN_LINE = "WHITE"
        self.pedestrian_started = False

        pkg_project_path = get_package_share_directory("one_1")
        self.model = YOLO(f"{pkg_project_path}/data/best_1.pt")

        self.next_goal = 0
        self.sleep = 1
    
    def pose_callback(self, data):
        self.pose = data

    def get_angle(self):
        quaternion = self.pose.pose.pose.orientation
        euler = euler_from_quaternion([quaternion.x, quaternion.y, quaternion.z, quaternion.w])
        return euler[2]

    def _warpPerspective(self, cvImg):
        h, w, _ = cvImg.shape

        pts1 = np.float32(
            [[0, 480], [w, 480], [50, 300], [w-50, 300]])
        result_img_width = np.int32(abs(pts1[0][0] - pts1[1][0]))
        result_img_height = np.int32(abs(pts1[0][1] - pts1[2][0]))

        pts2 = np.float32([[0, 0], [result_img_width, 0], [0, result_img_height], [
                          result_img_width, result_img_height]])

        M = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(
            cvImg, M, (result_img_width, result_img_height))

        return cv2.flip(dst, 0)

    def _find_yellow_line(self, perspectiveImg_, middle_h=None):
    
        perspectiveImg = perspectiveImg_[:, :perspectiveImg_.shape[1] // 2, :] if FOLLOW_ROAD_CROP_HALF else perspectiveImg_

        h, w, _ = perspectiveImg.shape
        middle_h = int(h * LINES_H_RATIO) if middle_h is None else middle_h

        yellow_mask = cv2.inRange(perspectiveImg, (0, 240, 255), (0, 255, 255))
        yellow_mask = cv2.dilate(yellow_mask, np.ones((2, 2)), iterations=4)
        middle_row = yellow_mask[middle_h]
       
        try:
            first_notYellow = np.int32(np.where(middle_row == 255))[0][-1]
            self._yellow_prevs.append(first_notYellow)
        except:  
            first_notYellow = sum(
                self._yellow_prevs)//len(self._yellow_prevs)
            self.point_status = False

        return first_notYellow

    def _find_white_line(self, perspectiveImg_, middle_h=None):

        perspectiveImg = perspectiveImg_[:, perspectiveImg_.shape[1] // 2:, :] if FOLLOW_ROAD_CROP_HALF else perspectiveImg_
        
        fix = perspectiveImg_.shape[1]//2 if FOLLOW_ROAD_CROP_HALF else 0

        h, w, _ = perspectiveImg.shape
        middle_h = int(h * LINES_H_RATIO) if middle_h is None else middle_h

        gray_img = cv2.cvtColor(perspectiveImg, cv2.COLOR_BGR2GRAY)
        white_mask = cv2.compare(gray_img, 250, cv2.CMP_GE)
        middle_row = white_mask[middle_h]
        
        try:
            first_white = np.int32(np.where(middle_row == 255))[0][0]
            self.__white_prevs.append(first_white)
        except:
            first_white = sum(self.__white_prevs)//len(self.__white_prevs)
            self.point_status = False

        return first_white + fix

    def _compute_PID(self, target):
        err = target
        e = np.arctan2(np.sin(err), np.cos(err))
        
        w = sum([self.Kp * e, self.Ki * (self.E + e), self.Kd * (e - self.prev_e)])
        w = np.arctan2(np.sin(w), np.cos(w))

        self.E += e
        self.prev_e = e
        return w

    def lidar_callback(self, data):
        self.lidar_data = data

    def _callback_camera(self, msg: Image):

        self.point_status = True

        twist = Twist()
        twist.linear.x = float(self._linear_speed)

        cvImg = self._cv_bridge.imgmsg_to_cv2(
            msg, desired_encoding=msg.encoding)
        cvImg = cv2.cvtColor(cvImg, cv2.COLOR_RGB2BGR)

        perspective = self._warpPerspective(cvImg)
        hLevelLine = int(perspective.shape[0]*LINES_H_RATIO)

        endYellow = WHITE_MODE_CONSTANT if self.MAIN_LINE == "WHITE" else self._find_yellow_line(perspective, hLevelLine)

        startWhite = YELLOW_MODE_CONSTANT if self.MAIN_LINE == "YELLOW" else self._find_white_line(perspective, hLevelLine) 

        middle_btw_lines = (startWhite + endYellow) // 2

        center = (perspective.shape[1]//2, hLevelLine)
        lines_center = (middle_btw_lines, hLevelLine)

        direction = center[0] - lines_center[0]

        if (abs(direction) > DIFF_CENTERS):
            angle_to_goal = math.atan2(
                direction, 215)
            angular_v = self._compute_PID(angle_to_goal)
            twist.angular.z = angular_v

            twist.linear.x = abs(self._linear_speed * (1 - abs(angular_v * (3 / 4 if ANALOG_CAP_MODE else 1))))

        if self.STATUS_CAR == 1 and self.avoidance == 0 and self.parking.state == 0:
            self._robot_cmd_vel_pub.publish(twist)

def main():
    rclpy.init()
    Trace = Follow_Trace_Node()
    rclpy.spin(Trace)
    Trace.destroy_node()
    rclpy.shutdown()