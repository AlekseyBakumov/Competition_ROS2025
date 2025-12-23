import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge


class TurnSignNode(Node):
    """Узел обнаружения синих поворотных знаков со стрелками.
    
    Детектирует круглые синие дорожные знаки с белыми стрелками, указывающими
    направление поворота (влево или вправо). Использует зрение на основе цвета (HSV)
    и глубины для фильтрации. Публикует команды управления (angular.z) для поворота робота.
    """
    
    def __init__(self):
        super().__init__('turn_sign_node')
        # Мост для конвертации между ROS Image и OpenCV форматами
        self.bridge = CvBridge()

        # Издатель команд поворота на основе обнаруженного знака
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel/turn_sign', 10)

        # Подписка на RGB-поток камеры для обнаружения знаков
        self.color_sub = self.create_subscription(Image, '/color/image', self.color_callback, 10)
        # Подписка на поток глубины для фильтрации на расстояние пересечения
        self.depth_sub = self.create_subscription(Image, '/depth/image', self.depth_callback, 10)
        # Издатель отладочного изображения (маски и контуры для визуализации)
        self.pub_dbg = self.create_publisher(Image, 'processed_sign/image', 10)
        
        # Буфер глубины (маска валидных пиксельных дистанций)
        self.img_dep = None
        # Флаг готовности кадра глубины
        self.img_dep_ready = False

        # Параметр: максимальная дистанция для обнаружения знаков (в метрах)
        self.declare_parameter('depth_camera_intersection_distance', 0.4)

        # Буфер RGB-изображения (в формате BGR из OpenCV)
        self.img_bgr = None
        # Главный цикл обработки с частотой 10 Гц
        self.timer = self.create_timer(0.1, self.loop)

        # Угловая скорость для поворота влево (положительное значение)
        self.left_angle = 0.9
        # Угловая скорость для поворота вправо (отрицательное значение)
        self.right_angle = -0.8
        # Линейная скорость при движении по команде знака (0 = стояние на месте)
        self.linear_speed = 0.0

        # Время удержания команды поворота после обнаружения знака (в секундах)
        self.command_duration = 0.5

        # Внутреннее время окончания активной команды (timestamp в сек)
        self._active_until = 0.0
        # Сохранённое направление: -1 левый, +1 правый, 0 нет команды
        self._last_dir = 0

        # Минимальная площадь синего контура для валидного обнаружения знака
        self.min_blue_area = 2000
        # Минимальная площадь белой стрелки для определения направления
        self.min_arrow_area = 400

    def color_callback(self, msg: Image):
        """Сохранить текущий RGB-кадр в буфер для обработки в главном цикле."""
        self.img_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def depth_callback(self, msg: Image):
        """Обработчик потока глубины. Создаёт маску пиксельных дистанций на пересечении.
        
        Фильтрует значения глубины, оставляя только пиксели, находящиеся на расстоянии
        пересечения дороги. Поддерживает два формата глубины (32FC1 и 16UC1).
        """
        try:
            # Конвертировать ROS Image в OpenCV формат
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding=msg.encoding)
        except Exception as e:
            print(f"CvBridge error: {e}")
            return

        # Получить параметр максимальной дистанции пересечения
        dist = self.get_parameter('depth_camera_intersection_distance').value
        
        # Обработка формата 32FC1 (32-битные float в метрах)
        if msg.encoding == '32FC1':
            # Маска: валидные (не inf/NaN), положительные и ближе порога
            self.img_dep = np.isfinite(cv_image) & (cv_image > 0) & (cv_image < dist)
            self.img_dep_ready = True
        # Обработка формата 16UC1 (16-битные unsigned int в миллиметрах)
        elif msg.encoding == '16UC1':
            # Маска: положительные пиксели в пределах дистанции
            self.img_dep = cv_image > 0 & cv_image < dist
            self.img_dep_ready = True
        else:
            print(f"Unsupported depth encoding: {msg.encoding}")
            return

    def loop(self):
        """Главный цикл управления. Управляет активацией и выполнением команд поворота.
        
        Вызывает детекцию знака, сохраняет направление если найдено,
        и публикует команды поворота с таймаутом (держит команду заданное время).
        """
        # Получить текущее время в секундах
        now = self.get_clock().now().nanoseconds / 1e9

        # Обнаружить направление знака (если он видим)
        direction = self.detect_turn_sign_direction(self.img_bgr)

        # Если знак обнаружен - обновить сохранённое направление и расширить время команды
        if direction is not None:
            self._last_dir = direction
            self._active_until = now + self.command_duration

        # Подготовить команду управления
        cmd = Twist()
        
        # Если команда активна (время не истекло) - отправить команду поворота
        if now < self._active_until and self._last_dir != 0:
            cmd.linear.x = float(self.linear_speed)
            # Выбрать угловую скорость в зависимости от направления (левый или правый)
            cmd.angular.z = float(self.left_angle if self._last_dir < 0 else self.right_angle)
        else:
            # Иначе - команда нейтральна (стояние на месте)
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

        # Опубликовать команду управления
        self.cmd_pub.publish(cmd)

    def detect_turn_sign_direction(self, bgr):
        """Детекция направления стрелки в синем дорожном знаке.
        
        Выполняет 3-этапный алгоритм обнаружения:
        ЭТАП 1: Обнаружить синий круг (фон знака) в HSV пространстве
        ЭТАП 2: Найти белую стрелку внутри синего круга
        ЭТАП 3: Определить направление стрелки по её положению относительно центра
        
        Args:
            bgr: RGB изображение в формате BGR (OpenCV)
            
        Returns:
            -1 если стрелка указывает влево
            +1 если стрелка указывает вправо
            None если знак не обнаружен
        """
        # Проверка наличия входных данных
        if bgr is None:
            return None
        if self.img_dep_ready is False:
            return None

        # Применить маску глубины к изображению (обнулить пиксели вне пересечения)
        bgr[~self.img_dep] = [0, 0, 0]
        
        # Конвертировать в HSV для работы с цветами
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # Опубликовать отладочное изображение
        msg_debug = self.bridge.cv2_to_imgmsg(bgr, encoding="passthrough")
        self.pub_dbg.publish(msg_debug)

        # ========== ЭТАП 1: Обнаружить синий круг (фон знака) ==========
        # Синий цвет в HSV: H примерно 90..130 (OpenCV использует H=0..179)
        blue_mask = cv2.inRange(hsv, (90, 80, 30), (130, 255, 255))
        # Морфологическое открытие (удаление шума)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        # Морфологическое закрытие (заполнение дыр внутри объектов)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

        # Найти все контуры синих объектов
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Выбрать самый большой синий контур (это должна быть сама окружность знака)
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        
        # Проверка минимальной площади
        if area < self.min_blue_area:
            return None

        # Проверка круглости объекта: циркулярность = 4π*площадь / периметр²
        # Для идеального круга = 1.0, для других фигур < 1.0
        peri = cv2.arcLength(c, True)
        if peri <= 1e-3:
            return None
        circularity = 4.0 * np.pi * area / (peri * peri)
        if circularity < 0.6:  # Объект должен быть достаточно круглым
            return None

        # Вычислить центр синего контура (центр знака)
        M = cv2.moments(c)
        if M["m00"] <= 1e-6:
            return None
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Создать маску области знака (залить синий контур белым)
        sign_mask = np.zeros(blue_mask.shape, dtype=np.uint8)
        cv2.drawContours(sign_mask, [c], -1, 255, -1)

        # ========== ЭТАП 2: Найти белую стрелку внутри синего круга ==========
        # Стрелка - светлая область: низкая насыщенность (S) и высокий Value (V)
        arrow_mask = cv2.inRange(hsv, (0, 0, 50), (179, 60, 255))
        # Применить маску знака - оставить стрелку только внутри синего круга
        arrow_mask = cv2.bitwise_and(arrow_mask, arrow_mask, mask=sign_mask)
        # Морфологическое открытие (удаление мелкого шума)
        arrow_mask = cv2.morphologyEx(arrow_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        # Морфологическое закрытие (заполнение дыр в стрелке)
        arrow_mask = cv2.morphologyEx(arrow_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

        # Найти все контуры светлых объектов (потенциальные стрелки)
        a_contours, _ = cv2.findContours(arrow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not a_contours:
            return None

        # Выбрать самый большой светлый контур (это должна быть стрелка)
        a = max(a_contours, key=cv2.contourArea)
        a_area = cv2.contourArea(a)
        
        # Проверка минимальной площади стрелки
        if a_area < self.min_arrow_area:
            return None

        # ========== ЭТАП 3: Определить направление стрелки ==========
        # Получить координаты всех точек контура стрелки
        pts = a.reshape(-1, 2)
        
        # Найти крайние X-координаты стрелки (левый и правый края)
        min_x = int(pts[:, 0].min())  # Самая левая точка стрелки
        max_x = int(pts[:, 0].max())  # Самая правая точка стрелки

        # Вычислить, насколько далеко стрелка простирается вправо и влево от центра
        right_extent = max_x - cx  # Расстояние от центра к правому краю
        left_extent = cx - min_x   # Расстояние от центра к левому краю

        # Если стрелка более выступает вправо - это стрелка вправо, иначе влево
        if right_extent > left_extent:
            return +1  # Стрелка вправо
        else:
            return -1  # Стрелка влево


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
