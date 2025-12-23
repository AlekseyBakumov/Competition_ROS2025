import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
import cv2
from cv_bridge import CvBridge

class ImageSubscriber(Node):
    """Узел обработки видеопотока с камеры для обнаружения дорожной разметки.
    
    Обрабатывает синхронизованные потоки RGB-изображений и глубины для выделения
    жёлтых и белых пиксельных линий (дорожная разметка) и генерирует сигнал
    коррекции трека (разница в пиксельных координатах между жёлтой и белой линиями).
    """
    
    def __init__(self):
        super().__init__('image_processor')
        # Подписка на поток RGB-изображений с камеры
        self.subscription_color = self.create_subscription(
            Image,
            'color/image',
            self.listener_callback,
            10)
        # Подписка на поток глубины (дистанция от датчика глубины)
        self.subscription_depth = self.create_subscription(
            Image,
            'depth/image',
            self.depth_callback,
            10)
        # Издатель обработанного изображения маски (для отладки/визуализации)
        self.publisherImg_ = self.create_publisher(Image, 'processed/image', 10)
        # Издатель сигнала коррекции трека (разница пикселей: жёлтые - белые)
        self.publisherTrack_ = self.create_publisher(Int32, 'track_lines/diff', 10)
        # Мост для конвертации между ROS Image и OpenCV cv2 форматами
        self.bridge = CvBridge()
        # Буфер для текущего кадра глубины (маска валидных пиксельных дистанций)
        self.depth_frame = None
        # Флаг готовности кадра глубины для синхронизации с RGB-потоком
        self.depth_frame_ready = False
        # Параметр: обрезка пикселей слева и справа (исключение краёв изображения)
        self.declare_parameter('color_camera_pixels_cutoff', 220)
        # Параметр: максимальная дистанция дорожных разметок (в метрах) для глубинной фильтрации
        self.declare_parameter('depth_camera_road_mark_distance', 0.25)

    def depth_callback(self, msg):
        """Обработчик потока глубины. Создаёт маску валидных пиксельных дистанций.
        
        Фильтрует значения глубины чтобы оставить только пиксели, которые находятся
        на дороге (от 0 до заданной дистанции). Поддерживает два формата глубины:
        - 32FC1: 32-битные float значения в метрах
        - 16UC1: 16-битные unsigned int значения в миллиметрах
        """
        try:
            # Конвертировать ROS Image сообщение в OpenCV формат
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding=msg.encoding)
        except Exception as e:
            print(f"CvBridge error: {e}")
            return

        # Получить параметр максимальной дистанции
        dist = self.get_parameter('depth_camera_road_mark_distance').value
        
        # Обработка глубины в формате 32-битных float (метры)
        if msg.encoding == '32FC1':
            # Маска: валидные (не inf/NaN), положительные и ближе чем порог дистанции
            self.depth_frame = np.isfinite(cv_image) & (cv_image > 0) & (cv_image < dist)
            self.depth_frame_ready = True
        # Обработка глубины в формате 16-битных unsigned int (миллиметры)
        elif msg.encoding == '16UC1':
            # Маска: положительные пиксели в пределах дистанции
            self.depth_frame = cv_image > 0 & cv_image < dist
            self.depth_frame_ready = True
        else:
            print(f"Unsupported depth encoding: {msg.encoding}")
            return

    def listener_callback(self, msg):
        """Основной обработчик RGB-потока. Синхронизирует с глубиной и обрабатывает изображение.
        
        Проверяет готовность кадра глубины для синхронизации, конвертирует цветовое
        пространство из BGR (формат OpenCV) в RGB (формат обработки) и передаёт
        изображение на основную обработку для выделения дорожной разметки.
        """
        # Проверить, готов ли кадр глубины для синхронизации
        if not self.depth_frame_ready:
            return

        # Конвертировать ROS Image в OpenCV изображение без изменения кодировки
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        # Конвертировать цветовое пространство: BGR (OpenCV) → RGB (стандартный формат)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        # Обработать изображение для поиска дорожной разметки
        self.process_track_lines(cv_image)


    def process_track_lines(self, image):
        """Основная функция обработки изображения для обнаружения дорожной разметки.
        
        Выполняет следующие этапы:
        1. Обрезка краёв изображения (исключение боков камеры)
        2. Синхронизация глубины с RGB изображением
        3. Маскирование изображения по глубине (оставляет только пиксели дороги)
        4. Подсчёт жёлтых и белых пикселей (линии разметки)
        5. Вычисление сигнала коррекции трека
        6. Публикация результатов
        """
        # Сообщение для публикации: целое число (разница пикселей)
        msg = Int32()
        
        # ЭТАП 1: Обрезать левый и правый края изображения
        cutoff = self.get_parameter('color_camera_pixels_cutoff').value
        image = image[:, cutoff:-cutoff]
        
        # ЭТАП 2: Обрезать маску глубины вместе с изображением
        depth_ = self.depth_frame.copy()
        depth_ = depth_[:, cutoff:-cutoff]

        # ЭТАП 3: Применить маску глубины к изображению (обнулить пиксели без дороги)
        masked_image = image.copy()
        masked_image[~depth_] = [0, 0, 0]  # Пиксели вне дороги → чёрные
        
        # ЭТАП 4: Подсчитать жёлтые и белые пиксели (дорожные линии)
        mask_y, mask_w, diff = self.count_yellow_white_pixels(masked_image)

        # ЭТАП 5: Формировать сигнал трека: разница между жёлтыми и белыми пикселями
        # Положительное значение → белая линия справа
        # Отрицательное значение → жёлтая линия слева
        msg.data = diff
        self.publisherTrack_.publish(msg)
        
        # ЭТАП 6: Опубликовать маску белых пикселей для отладки/визуализации
        image_message = self.bridge.cv2_to_imgmsg(mask_w, encoding="passthrough")
        self.publisherImg_.publish(image_message)

    def count_yellow_white_pixels(self, image):
        """Подсчитать жёлтые и белые пиксели в изображении.
        
        Использует HSV-подобные пороги в RGB пространстве для выделения линий дорожной
        разметки. Жёлтая линия обычно справа, белая линия обычно слева.
        
        Args:
            image: RGB изображение, обработанное маской глубины
            
        Returns:
            tuple: (yellow_mask, white_mask, diff)
            - yellow_mask: бинарная маска жёлтых пикселей
            - white_mask: бинарная маска белых пикселей
            - diff: разница (жёлтые - белые) для сигнала коррекции трека
        """
        # Уже в RGB формате (из listener_callback), но переконвертируем для надёжности
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Диапазон цвета для жёлтой разметки (RGB)
        # Жёлтый: высокие R, высокие G, низкий B
        yellow_lower = np.array([200, 200, 0])
        yellow_upper = np.array([255, 255, 20])
        
        # Диапазон цвета для белой разметки (RGB)
        # Белый: высокие R, высокие G, высокие B (все компоненты близки)
        white_lower = np.array([230, 230, 230])
        white_upper = np.array([255, 255, 255])
        
        # Создать бинарные маски: пиксели в диапазоне → 255, остальные → 0
        yellow_mask = cv2.inRange(rgb, yellow_lower, yellow_upper)
        white_mask = cv2.inRange(rgb, white_lower, white_upper)
        
        # Подсчитать ненулевые пиксели (количество жёлтых и белых)
        yellow_count = np.count_nonzero(yellow_mask)
        white_count = np.count_nonzero(white_mask)
        
        # Вычислить сигнал коррекции трека
        # Положительный diff → больше жёлтых пикселей → нужен поворот налево
        # Отрицательный diff → больше белых пикселей → нужен поворот направо
        diff = yellow_count - white_count
        
        return yellow_mask, white_mask, diff




def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    rclpy.spin(image_subscriber)
    image_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
