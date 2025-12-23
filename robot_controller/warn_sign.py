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
        # ROS-OpenCV конвертор
        self.bridge = CvBridge()

        # Подписка на цветное и глубинное изображение с камеры
        self.sub = self.create_subscription(Image, 'color/image', self.cb_img, 10)
        self.depth_sub = self.create_subscription(Image, 'depth/image', self.depth_callback, 10)

        # Издатель для результата детекции (строка "FOUND" или "NONE")
        self.pub = self.create_publisher(String, 'sign/construction', 10)
        # Издатель отладочного изображения с отмеченными знаками
        self.pub_dbg = self.create_publisher(Image, 'sign/construction/image', 10)

        # Пороги детекции для фильтрации ложных срабатываний
        self.min_triangle_area = 1000    # Минимальная площадь треугольника
        self.min_red_ratio = 0.06        # Минимум 6% красных пикселей внутри знака
        self.min_yellow_ratio = 0.18     # Минимум 18% жёлтых пикселей (полоса предупреждения)
        self.min_black_ratio = 0.02      # Минимум 2% чёрных пикселей (разметка)

        # Максимальная дистанция для детекции строительного знака (метры)
        self.dist = 0.4

        # Буфер изображения глубины и флаг готовности
        self.img_dep = None
        self.img_dep_ready = False

        # Буфер цветного изображения
        self.last_bgr = None
        # Главный цикл обработки 10 Гц
        self.timer = self.create_timer(0.1, self.loop)

    def depth_callback(self, msg: Image):
        # Обработать карту глубины для фильтрации объектов на нужной дистанции
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding=msg.encoding)
        except Exception as e:
            self.get_logger().warn(f"Depth CvBridge error: {e}")
            self.img_dep_ready = False
            self.img_dep = None
            return

        if msg.encoding == '32FC1':
            # 32-битное число с плавающей точкой: значения в метрах
            self.img_dep = np.isfinite(depth) & (depth > 0.0) & (depth < float(self.dist))
            self.img_dep_ready = True

        elif msg.encoding == '16UC1':
            # 16-битное беззнаковое целое число: обычно в миллиметрах
            dist_mm = int(float(self.dist) * 1000.0)
            self.img_dep = (depth > 0) & (depth < dist_mm)
            self.img_dep_ready = True

        else:
            self.get_logger().warn(f"Unsupported depth encoding: {msg.encoding}")
            self.img_dep_ready = False
            self.img_dep = None

    def cb_img(self, msg: Image):
        # Сохранить цветное изображение в буфер
        try:
            self.last_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f"Color CvBridge error: {e}")
            self.last_bgr = None

    def loop(self):
        # Главный цикл: детекция треугольных строительных знаков (красно-жёлто-чёрные)
        # Всегда публикуем результат, даже если ничего не найдено
        out = String()
        out.data = "NONE"

        # Если нет цветного изображения, отправить результат "не найдено"
        if self.last_bgr is None:
            self.pub.publish(out)
            return

        bgr = self.last_bgr.copy()

        # Применить маску глубины для фильтрации объектов по дистанции
        if self.img_dep_ready and self.img_dep is not None:
            if self.img_dep.shape[:2] == bgr.shape[:2]:
                # Обнулить пиксели, которые находятся далеко от камеры
                bgr[~self.img_dep] = (0, 0, 0)
            else:
                # Если размеры не совпадают, продолжить без маски глубины
                self.get_logger().warn_throttle(2.0, "Depth size != color size, ignoring depth mask")

        # Преобразовать в HSV для цветовой обработки
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # ЭТАП 1: Детекция красного цвета (основной цвет знака)
        # Красный цвет имеет две области в HSV (H оборачивается вокруг 0/179)
        red1 = cv2.inRange(hsv, (0, 80, 80), (10, 255, 255))       # Нижняя область красного
        red2 = cv2.inRange(hsv, (160, 80, 80), (179, 255, 255))    # Верхняя область красного
        red_mask = cv2.bitwise_or(red1, red2)

        # Морфологическая обработка для удаления шума
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

        # Найти все красные контуры (потенциальные знаки)
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Подготовить отладочное изображение
        dbg = bgr.copy()
        found = False

        # Проанализировать все красные контуры
        for c in contours:
            area = cv2.contourArea(c)
            # Отфильтровать по минимальному размеру
            if area < self.min_triangle_area:
                continue

            # ЭТАП 2: Проверка формы (должно быть треугольником - 3 стороны)
            peri = cv2.arcLength(c, True)
            # Аппроксимировать контур к многоугольнику
            approx = cv2.approxPolyDP(c, 0.03 * peri, True)

            # Проверить, что это именно треугольник (3 вершины)
            if len(approx) != 3:
                continue

            # Получить ограничивающий прямоугольник и проверить размер
            x, y, w, h = cv2.boundingRect(approx)
            if w < 40 or h < 40:
                continue

            # ЭТАП 3: Проверка цветовой композиции внутри треугольника
            # Извлечь область знака
            roi_hsv = hsv[y:y + h, x:x + w]
            roi_red = red_mask[y:y + h, x:x + w]

            # Вычислить площадь области
            roi_area = float(w * h)
            
            # Проверить количество красных пикселей
            red_ratio = float(np.count_nonzero(roi_red)) / max(1.0, roi_area)
            if red_ratio < self.min_red_ratio:
                continue

            # Проверить наличие жёлтой полосы (часть предупредительного знака)
            yellow_mask = cv2.inRange(roi_hsv, (15, 70, 70), (40, 255, 255))
            yellow_ratio = float(np.count_nonzero(yellow_mask)) / max(1.0, roi_area)
            if yellow_ratio < self.min_yellow_ratio:
                continue

            # Проверить наличие чёрной разметки
            black_mask = cv2.inRange(roi_hsv, (0, 0, 0), (179, 255, 60))
            black_ratio = float(np.count_nonzero(black_mask)) / max(1.0, roi_area)
            if black_ratio < self.min_black_ratio:
                continue

            # ЗНАК НАЙДЕН! Знак имеет все признаки: красный треугольник с жёлтой и чёрной разметкой
            found = True
            # Нарисовать контур найденного знака
            cv2.drawContours(dbg, [approx], -1, (0, 255, 0), 3)
            cv2.putText(dbg, "CONSTRUCTION SIGN", (x, max(0, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            break

        # Опубликовать результат детекции
        out.data = "FOUND" if found else "NONE"
        self.pub.publish(out)

        # Опубликовать отладочное изображение с результатами
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
    