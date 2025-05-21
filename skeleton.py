# !/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Header

class DetermineColor(Node):
    def __init__(self):
        super().__init__('color_detector')
        self.image_sub = self.create_subscription(Image, '/camera/color/image_raw', self.callback, 10)
        self.color_pub = self.create_publisher(Header, '/rotate_cmd', 10)
        self.bridge = CvBridge()
        self.count = 0

    def callback(self, data):
        try:
            # listen image topic
            image = self.bridge.imgmsg_to_cv2(data, 'bgr8')

            # prepare rotate_cmd msg
            # DO NOT DELETE THE BELOW THREE LINES!
            msg = Header()
            msg = data.header
            msg.frame_id = '0'  # default: STOP
            
            cv2.imshow('Image',image)
            cv2.waitKey(1)
            
            # determine background color
            # TODO 
            # determine the color and assing +1, 0, or, -1 for frame_id
            # msg.frame_id = '+1' # CCW 
            # msg.frame_id = '0'  # STOP
            # msg.frame_id = '-1' # CW 
            
            red_intensity=image[20, 10, 2]
            blue_intensity=image[20, 10, 0]
            if red_intensity > blue_intensity:
            	msg.frame_id = '-1'
            else:
            	msg.frame_id = '+1'
            self.count+=1
            #if self.count > 300 and self.count < 600:
            #	msg.frame_id = '+1'
            #elif self.count > 600 and self.count < 900:
            #	msg.frame_id = '-1'
            #elif self.count > 900 and self.count <1200:
            #	msg.frame_id = '0'
            #elif self.count > 1200 and self.count < 1500:
            #	msg.frame_id = '+1'
            #elif self.count > 1500 and self.count < 1800:
            #	msg.frame_id = '-1'
            #elif self.count > 1800 and self.count < 2100:
            #	msg.frame_id = '0'
            
            # publish color_state
            self.color_pub.publish(msg)
        except CvBridgeError as e:
            self.get_logger().error('Failed to convert image: %s' % e)


if __name__ == '__main__':
    rclpy.init()
    detector = DetermineColor()
    rclpy.spin(detector)
    detector.destroy_node()
    rclpy.shutdown()

