import rclpy
from rclpy.node import Node

import cv2
import numpy as np

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge


class SvetoforSimple(Node):
    def __init__(self):
        super().__init__('svetofor_simple')

        # ROS-OpenCV конвертор
        self.bridge = CvBridge()

        # Издатель команды движения при детекции зелёного света
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel/svetofor', 10)
        # Издатель для отладочного изображения с результатами детекции
        self.pub_dbg = self.create_publisher(Image, 'processed_traffic/image', 10)

        # Подписка на цветное изображение с камеры
        self.sub = self.create_subscription(Image, '/color/image', self.cb_img, 10)

        # Параметры детекции светофора
        self.declare_parameter('move_speed', 1.0)

        # Диапазон HSV для зелёного цвета (OpenCV H: 0..179)
        self.declare_parameter('h_low', 45)
        self.declare_parameter('h_high', 95)
        self.declare_parameter('s_low', 60)     # Минимальная насыщенность
        self.declare_parameter('v_low', 60)     # Минимальная яркость

        # Параметры фильтрации контуров
        self.declare_parameter('min_area', 80)          # Минимальная площадь контура
        self.declare_parameter('min_circularity', 0.55) # 1.0 = идеальный круг

        # Область интереса (ROI): только верхняя часть кадра для светофоров
        # 1.0 = весь кадр, 0.5 = верхняя половина
        self.declare_parameter('roi_top_fraction', 1.0)

        # Буфер текущего изображения
        self.last_bgr = None
        # Главный цикл обработки 10 Гц
        self.timer = self.create_timer(0.1, self.loop)

    def cb_img(self, msg: Image):
        # Сохранить цветное изображение в буфер для обработки
        try:
            self.last_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception:
            self.last_bgr = None

    def loop(self):
        # Главный цикл: детекция зелёного светофора и выдача команды движения
        cmd = Twist()
        cmd.linear.x = 0.0  # По умолчанию не движемся
        cmd.angular.z = 0.0

        # Если нет изображения, отправить команду стоп
        if self.last_bgr is None:
            self.pub_cmd.publish(cmd)
            return

        bgr = self.last_bgr
        h, w = bgr.shape[:2]

        # Извлечь область интереса: верхняя часть кадра (где обычно находится светофор)
        roi_frac = float(self.get_parameter('roi_top_fraction').value)
        roi_frac = max(0.1, min(1.0, roi_frac))
        y1 = int(h * roi_frac)
        roi = bgr[0:y1, :]

        # Преобразовать в HSV для цветовой обработки
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Получить параметры HSV для зелёного цвета
        h_low = int(self.get_parameter('h_low').value)
        h_high = int(self.get_parameter('h_high').value)
        s_low = int(self.get_parameter('s_low').value)
        v_low = int(self.get_parameter('v_low').value)

        # Создать маску для зелёного цвета по заданному диапазону HSV
        mask = cv2.inRange(hsv, (h_low, s_low, v_low), (h_high, 255, 255))

        # Морфологическая обработка для удаления шума
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

        # Найти контуры в маске
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        found = False
        best = None
        best_area = 0.0

        # Получить пороги для фильтрации контуров
        min_area = float(self.get_parameter('min_area').value)
        min_circ = float(self.get_parameter('min_circularity').value)

        # Проанализировать все контуры и найти зелёный круг
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            # Вычислить коэффициент округлости (1.0 = идеальный круг)
            peri = cv2.arcLength(c, True)
            if peri < 1e-3:
                continue
            circularity = 4.0 * np.pi * area / (peri * peri)
            # Отфильтровать объекты, которые не похожи на круги
            if circularity < min_circ:
                continue
            # Выбрать самый большой круг
            if area > best_area:
                best_area = area
                best = c

        # Если найден зелёный круг, разрешить движение
        if best is not None:
            found = True
            cmd.linear.x = float(self.get_parameter('move_speed').value)

        # Опубликовать команду управления
        self.pub_cmd.publish(cmd)

        # Подготовить и опубликовать отладочное изображение с результатами детекции
        dbg = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        if best is not None:
            # Нарисовать контур найденного зелёного круга
            cv2.drawContours(dbg, [best], -1, (0, 0, 255), 2)
            cv2.putText(dbg, f"GREEN CIRCLE! area={best_area:.0f}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(dbg, "NO GREEN CIRCLE",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        self.pub_dbg.publish(self.bridge.cv2_to_imgmsg(dbg, encoding='bgr8'))


def main(args=None):
    rclpy.init(args=args)
    node = SvetoforSimple()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
