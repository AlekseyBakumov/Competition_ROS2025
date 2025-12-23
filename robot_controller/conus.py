import rclpy
from rclpy.node import Node

import numpy as np

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist


class ConeAvoiderLidar(Node):
    def __init__(self):
        super().__init__('cone_avoider_lidar')

        # Подписка на сканы LiDAR
        self.sub = self.create_subscription(LaserScan, '/scan', self.cb_scan, 10)
        # Издатель команд управления для движения с избеганием препятствий
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Буфер для текущего скана LiDAR
        self.scan = None
        # Главный цикл обработки с частотой 20 Гц
        self.timer = self.create_timer(0.05, self.loop)

        # ---- ПАРАМЕТРЫ ИЗБЕГАНИЯ ПРЕПЯТСТВИЙ ----
        # Поле зрения (FOV): общий сектор для анализа (±70° = 140°)
        self.declare_parameter('front_fov_deg', 140.0)
        # Дистанция порога обнаружения: ближе = препятствие
        self.declare_parameter('obstacle_dist', 0.85)
        # Минимальная валидная дистанция для фильтрации шума и артефактов
        self.declare_parameter('min_valid', 0.05)
        # Радиус робота в метрах для расчёта безопасной зоны
        self.declare_parameter('robot_radius', 0.18)
        # Дополнительный запас безопасности сверх радиуса робота
        self.declare_parameter('safety_margin', 0.08)
        # Максимальная скорость движения вперёд
        self.declare_parameter('max_speed', 0.25)
        # Минимальная скорость движения (при близких препятствиях)
        self.declare_parameter('min_speed', 0.08)
        # Максимальная угловая скорость поворота
        self.declare_parameter('max_angular', 1.6)
        # Коэффициент усиления рулевого управления: как сильно поворачиваем
        self.declare_parameter('k_steer', 1.4)
        # Предпочтение для прямого движения (0..1): штраф за отклонение от центра
        self.declare_parameter('prefer_straight', 1.0)

    def cb_scan(self, msg: LaserScan):
        # Сохранить новый скан LiDAR для обработки в главном цикле
        self.scan = msg

    def loop(self):
        # Главный цикл избегания препятствий на основе gap-finding алгоритма
        # Находит свободные промежутки между препятствиями и выбирает наилучший для прохода
        if self.scan is None:
            return

        scan = self.scan
        # Преобразовать массив дистанций в numpy array для удобной обработки
        ranges = np.array(scan.ranges, dtype=np.float32)

        # Обработка инвалидных значений: заменить inf/NaN на максимальное значение
        max_r = scan.range_max if scan.range_max > 0 else 10.0
        ranges = np.where(np.isfinite(ranges), ranges, max_r)

        # Вычислить угол для каждого луча LiDAR
        n = ranges.shape[0]
        angles = scan.angle_min + np.arange(n, dtype=np.float32) * scan.angle_increment

        # Выбрать только фронтальный сектор (пропустить лучи слева и справа)
        fov = np.deg2rad(float(self.get_parameter('front_fov_deg').value))
        half = 0.5 * fov
        front_mask = (angles >= -half) & (angles <= half)

        # Получить только фронтальные дистанции и углы
        r = ranges[front_mask]
        a = angles[front_mask]

        # Отфильтровать и обрезать дистанции до валидного диапазона
        min_valid = float(self.get_parameter('min_valid').value)
        r = np.clip(r, min_valid, max_r)

        # ЭТАП 1: Обнаружить базовые препятствия по порогу дистанции
        obstacle_dist = float(self.get_parameter('obstacle_dist').value)
        occupied = r < obstacle_dist

        # ЭТАП 2: Инфляция препятствий по габаритам робота и запасу безопасности
        # Расширить зону препятствий с учётом физических размеров робота
        robot_radius = float(self.get_parameter('robot_radius').value)
        safety = float(self.get_parameter('safety_margin').value)
        inflate_radius = robot_radius + safety

        occupied_infl = occupied.copy()
        idx_occ = np.where(occupied)[0]
        if idx_occ.size > 0:
            # Для каждого обнаруженного препятствия расширить его зону
            for i in idx_occ:
                dist = float(r[i])
                # Вычислить угловую ширину инфляции
                ang_infl = np.arctan2(inflate_radius, max(dist, 1e-3))
                # Сколько индексов луча это покрывает
                k = int(np.ceil(ang_infl / abs(scan.angle_increment)))
                # Расширить препятствие в обе стороны
                i0 = max(0, i - k)
                i1 = min(len(occupied_infl) - 1, i + k)
                occupied_infl[i0:i1 + 1] = True

        # Вычислить свободные области (инверсия занятых)
        free = ~occupied_infl

        # Если всё вокруг заблокировано - крутиться на месте
        if not np.any(free):
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = float(self.get_parameter('max_angular').value) * 0.6
            self.pub.publish(cmd)
            return

        # ЭТАП 3: Найти все свободные промежутки (gaps) между препятствиями
        # Определить начало и конец каждого свободного сегмента
        free_int = free.astype(np.int32)
        changes = np.diff(free_int)
        # Индексы, где свобода начинается (переход от занятого к свободному)
        starts = list(np.where(changes == 1)[0] + 1)
        # Индексы, где свобода заканчивается (переход от свободного к занятому)
        ends = list(np.where(changes == -1)[0])

        # Обработать краевые случаи (если свобода начинается/заканчивается у края)
        if free[0]:
            starts = [0] + starts
        if free[-1]:
            ends = ends + [len(free) - 1]

        # Составить список всех gap-ов (начало, конец)
        gaps = list(zip(starts, ends))
        if not gaps:
            # Нет свободных промежутков - остановиться
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.pub.publish(cmd)
            return

        # ЭТАП 4: Выбрать наилучший gap
        # Оценить каждый gap по двум критериям: ширина + близость к прямому направлению
        prefer_straight = float(self.get_parameter('prefer_straight').value)

        best_score = -1e9
        best_idx = None
        for s, e in gaps:
            # Ширина gap-а (количество лучей)
            width = e - s + 1
            # Середина gap-а (индекс центрального луча)
            mid = (s + e) // 2
            # Угол центрального луча
            ang_mid = float(a[mid])

            # Оценка: чем шире gap - тем лучше, чем ближе к 0° - тем лучше
            score = width - prefer_straight * (abs(ang_mid) / half) * 80.0
            if score > best_score:
                best_score = score
                best_idx = mid

        # Получить целевой угол (направление лучшего gap-а)
        target_angle = float(a[best_idx])

        # ЭТАП 5: Управление - вычислить скорость и угловую скорость
        # Коэффициент усиления для рулевого управления
        k_steer = float(self.get_parameter('k_steer').value)
        max_ang = float(self.get_parameter('max_angular').value)

        # Угловая скорость: пропорциональна углу до лучшего gap-а, ограничена max_ang
        ang_z = np.clip(k_steer * target_angle, -max_ang, max_ang)

        # Линейная скорость: адаптивная в зависимости от ближайшего препятствия впереди
        # Берём минимальную дистанцию в узком фронтальном секторе (±15°)
        narrow = (a > -np.deg2rad(15)) & (a < np.deg2rad(15))
        front_min = float(np.min(r[narrow])) if np.any(narrow) else float(np.min(r))

        v_max = float(self.get_parameter('max_speed').value)
        v_min = float(self.get_parameter('min_speed').value)

        # Линейная интерполяция скорости: при obstacle_dist → v_min, при (obstacle_dist+0.8) → v_max
        v = v_min + (v_max - v_min) * np.clip((front_min - obstacle_dist) / 0.8, 0.0, 1.0)

        # Если сильно поворачиваем - снизить скорость для безопасности
        v *= (1.0 - 0.35 * min(1.0, abs(ang_z) / max_ang))

        # Составить и отправить команду управления
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
