import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError

import cv2
import numpy as np
import random

class DetermineColor(Node):
    def __init__(self):
        super().__init__('color_detector')
        self.image_sub = self.create_subscription(Image, '/color', self.callback, 10)
        self.color_pub = self.create_publisher(Header, '/rotate_cmd', 10)
        self.bridge = CvBridge()

    def callback(self, data):
        try:
            # ROS 이미지 → OpenCV 이미지
            image = self.bridge.imgmsg_to_cv2(data, 'bgr8')

            # 기본 메시지 설정
            msg = Header()
            msg = data.header
            msg.frame_id = '0' #기본 값

            # 색 판단
            result = self.detect_color(image)

            # 결과에 따라 frame_id 결정
            if result == 'R':
                msg.frame_id = '-1'  # CW
            elif result == 'B':
                msg.frame_id = '+1'  # CCW
            else:
                msg.frame_id = '0'   # Unknown일 경우 STOP

            self.color_pub.publish(msg)
        except CvBridgeError as e:
            self.get_logger().error('Failed to convert image: %s' % e)

    def detect_color(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask_black = cv2.inRange(hsv, (0, 0, 0), (180, 255, 40)) #검은색 모니터 인식 기준
        contours, hierarchy = cv2.findContours(mask_black, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        if not contours or hierarchy is None:
            return 'Unknown'

        hierarchy = hierarchy[0]
        total = len(contours)
        pick_count = int(total * 0.7) #조사하는 컨투어의 비율

        step = max(1, total // pick_count)
        candidate_indices = list(range(0, total, step))
        selected_indices = random.sample(candidate_indices, min(pick_count, len(candidate_indices)))

        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for i in selected_indices:
            parent = hierarchy[i][3]
            if parent == -1 or (parent != -1 and hierarchy[parent][3] == -1):
                cv2.drawContours(mask, contours, i, 255, -1)

        roi = hsv[mask == 255]
        if roi.size == 0:
            return 'Unknown'

        h, s, v = roi[:, 0], roi[:, 1], roi[:, 2]
        valid = (v > 35) & (s > 50)

        count_r = np.count_nonzero(((h > 160) | (h < 20)) & valid)
        count_b = np.count_nonzero((h > 100) & (h < 140) & valid)

        if count_r == count_b == 0:
            return 'Unknown'

        if count_r > count_b:
            return 'R'
        elif count_r < count_b:
            return 'B'
        else:
            return 'Unknown'

if __name__ == '__main__':
    rclpy.init()
    detector = DetermineColor()
    rclpy.spin(detector)
    detector.destroy_node()
    rclpy.shutdown()
