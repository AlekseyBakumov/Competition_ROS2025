import rclpy
from rclpy.node import Node

import numpy as np

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist


class ConeAvoiderLidar(Node):
    def __init__(self):
        super().__init__('cone_avoider_lidar')

        self.sub = self.create_subscription(LaserScan, '/scan', self.cb_scan, 10)
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.scan = None
        self.timer = self.create_timer(0.05, self.loop)  # 20 Hz

        # ---- params ----
        self.declare_parameter('front_fov_deg', 140.0)          # общий сектор (±70)
        self.declare_parameter('obstacle_dist', 0.85)           # ближе => препятствие
        self.declare_parameter('min_valid', 0.05)               # фильтр мусора
        self.declare_parameter('robot_radius', 0.18)            # радиус робота (м)
        self.declare_parameter('safety_margin', 0.08)           # запас (м)
        self.declare_parameter('max_speed', 0.25)
        self.declare_parameter('min_speed', 0.08)
        self.declare_parameter('max_angular', 1.6)
        self.declare_parameter('k_steer', 1.4)                  # усиление руля
        self.declare_parameter('prefer_straight', 1.0)          # 0..1, штраф за отклонение от 0

    def cb_scan(self, msg: LaserScan):
        self.scan = msg

    def loop(self):
        if self.scan is None:
            return

        scan = self.scan
        ranges = np.array(scan.ranges, dtype=np.float32)

        # заменим inf/NaN
        max_r = scan.range_max if scan.range_max > 0 else 10.0
        ranges = np.where(np.isfinite(ranges), ranges, max_r)

        # углы для каждого луча
        n = ranges.shape[0]
        angles = scan.angle_min + np.arange(n, dtype=np.float32) * scan.angle_increment

        # берём фронтальный сектор
        fov = np.deg2rad(float(self.get_parameter('front_fov_deg').value))
        half = 0.5 * fov
        front_mask = (angles >= -half) & (angles <= half)

        r = ranges[front_mask]
        a = angles[front_mask]

        # фильтр “слишком близких” и слишком дальних
        min_valid = float(self.get_parameter('min_valid').value)
        r = np.clip(r, min_valid, max_r)

        # 1) базовая занятость по порогу
        obstacle_dist = float(self.get_parameter('obstacle_dist').value)
        occupied = r < obstacle_dist

        # 2) инфляция препятствий по габаритам
        # угол инфляции примерно: atan((robot_radius+safety)/dist)
        robot_radius = float(self.get_parameter('robot_radius').value)
        safety = float(self.get_parameter('safety_margin').value)
        inflate_radius = robot_radius + safety

        occupied_infl = occupied.copy()
        idx_occ = np.where(occupied)[0]
        if idx_occ.size > 0:
            for i in idx_occ:
                dist = float(r[i])
                ang_infl = np.arctan2(inflate_radius, max(dist, 1e-3))
                # сколько индексов это покрывает
                k = int(np.ceil(ang_infl / abs(scan.angle_increment)))
                i0 = max(0, i - k)
                i1 = min(len(occupied_infl) - 1, i + k)
                occupied_infl[i0:i1 + 1] = True

        free = ~occupied_infl

        # если всё занято — стоп/поворот на месте
        if not np.any(free):
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = float(self.get_parameter('max_angular').value) * 0.6
            self.pub.publish(cmd)
            return

        # 3) найти все free-segments (gaps)
        # индексы начала/конца
        free_int = free.astype(np.int32)
        changes = np.diff(free_int)
        starts = list(np.where(changes == 1)[0] + 1)
        ends = list(np.where(changes == -1)[0])

        if free[0]:
            starts = [0] + starts
        if free[-1]:
            ends = ends + [len(free) - 1]

        gaps = list(zip(starts, ends))
        if not gaps:
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.pub.publish(cmd)
            return

        # 4) выбрать лучший gap: широкий + ближе к 0°
        prefer_straight = float(self.get_parameter('prefer_straight').value)

        best_score = -1e9
        best_idx = None
        for s, e in gaps:
            width = e - s + 1
            mid = (s + e) // 2
            ang_mid = float(a[mid])

            # чем шире gap — тем лучше
            # чем ближе к 0 — тем лучше
            score = width - prefer_straight * (abs(ang_mid) / half) * 80.0
            if score > best_score:
                best_score = score
                best_idx = mid

        target_angle = float(a[best_idx])

        # 5) управление
        k_steer = float(self.get_parameter('k_steer').value)
        max_ang = float(self.get_parameter('max_angular').value)

        ang_z = np.clip(k_steer * target_angle, -max_ang, max_ang)

        # скорость: чем ближе препятствие впереди, тем медленнее
        # берем минимальную дистанцию в узком секторе ±15°
        narrow = (a > -np.deg2rad(15)) & (a < np.deg2rad(15))
        front_min = float(np.min(r[narrow])) if np.any(narrow) else float(np.min(r))

        v_max = float(self.get_parameter('max_speed').value)
        v_min = float(self.get_parameter('min_speed').value)

        # линейная шкала: obstacle_dist..(obstacle_dist+0.8)
        v = v_min + (v_max - v_min) * np.clip((front_min - obstacle_dist) / 0.8, 0.0, 1.0)

        # если сильно крутим — чуть снизим скорость
        v *= (1.0 - 0.35 * min(1.0, abs(ang_z) / max_ang))

        cmd = Twist()
        cmd.linear.x = float(v)
        cmd.angular.z = float(ang_z)
        self.pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = ConeAvoiderLidar()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
