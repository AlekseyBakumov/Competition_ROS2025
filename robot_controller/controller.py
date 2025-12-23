import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from std_msgs.msg import Int32

class Controller(Node):
    def __init__(self):
        super().__init__('vehicle_controller')
        self.subscription = self.create_subscription(Int32, 'track_lines/diff', self.listener_callback, 10)

        self.sub_tl = self.create_subscription(
            Twist, '/cmd_vel/svetofor', self.cb_traffic, 10
        )

        self.sub_sign = self.create_subscription(
            Twist, '/cmd_vel/turn_sign', self.cb_sign, 10
        )
        
        self.red_sign_sub = self.create_subscription(
            String, '/sign/construction', self.constr_sign, 10
        )

        self.allowed_to_move = False

        self.publisher_cmd_vel_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer = self.create_timer(0.1, self.loop)
        self.current_diff = 0

        self.sign_timeout = 1.0

        self.k_diff = 1/5000

        self.base_speed = 0.10
        self.sign_speed = 0.10

        self.found_blue_sign = False

        self.warn_sign = False


    def listener_callback(self, msg):
        self.current_diff = msg.data


    def cb_sign(self, msg):
        self.last_sign_cmd = msg
        self.last_sign_time = self.get_clock().now()

    
    def cb_traffic(self, msg: Twist):
        self.last_tl_time = self.get_clock().now()
        if msg.linear.x > 0.0:
            self.allowed_to_move = True

    def constr_sign(self, msg):
        self.warn_sign = (msg.data == "FOUND")

    def loop(self):
        msg = Twist()

        now = self.get_clock().now()

        msg.linear.x = float(self.base_speed)
        msg.linear.y = 0.0
        msg.linear.z = 0.0
        
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = -(self.current_diff / 5000)
        
        # светофор
        if not self.allowed_to_move:
            msg.linear.x = 0.0
            msg.angular.z = 0.0
            self.publisher_cmd_vel_.publish(msg)
            return
        
        # знак
        sign_active = False
        if self.last_sign_time is not None:
            age = (now - self.last_sign_time).nanoseconds / 1e9
            sign_active = age < self.sign_timeout and (
                abs(self.last_sign_cmd.angular.z) > 1e-3 or abs(self.last_sign_cmd.linear.x) > 1e-3
            )
        if sign_active:
            msg.linear.x = float(self.sign_speed if self.sign_speed is not None else self.base_speed)
            msg.angular.z = float(self.last_sign_cmd.angular.z)
        else:
            msg.linear.x = float(self.base_speed)
            msg.angular.z = -float(self.current_diff) * float(self.k_diff)
        
        if self.last_sign_cmd.angular.z != 0:
            self.found_blue_sign = True

        if self.found_blue_sign and self.allowed_to_move and self.warn_sign:
            msg.linear.x = 0.0
            msg.angular.z = 0.0
            self.publisher_cmd_vel_.publish(msg)
            self.finish()
            return

        self.publisher_cmd_vel_.publish(msg)
        
    def finish(self):
        msg = String()
        self.pub = self.create_publisher(String, '/robot_finish', 10)
        msg.data = "NAVI: Копылов Матвей, Бакумов Алексей, Тишкин Андрей"
        self.pub.publish(msg)
        return
        


def main(args=None):
    rclpy.init(args=args)
    cmd_vel_publisher = Controller()
    rclpy.spin(cmd_vel_publisher)
    cmd_vel_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

    