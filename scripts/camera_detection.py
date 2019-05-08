#!/usr/bin/env python3

# ============================ Information =============================
# Project:      Texas A&M AutoDrive Challenge - Year 2
# Language:     Python 3.5.2
# ROS Package:  camera_detection
# File Name:    camera_detection.py
# Version:      1.0.0
# Description:  Provide access to Neural Networks through ROS.
# Date:         September 24, 2018

# ============================== Imports ===============================
import cv2
import signal
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import PIL
from PIL import ImageFile
from PIL import Image as PILImage
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Float32
from std_msgs.msg import Header
ImageFile.LOAD_TRUNCATED_IMAGES = True
from std_msgs.msg import String
# neural network
import tensorflow as tf
import keras
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from path_planner.msg import TrafficLightArray, TrafficLight, TrafficSignArray, TrafficSign, ObstacleArray, Obstacle
import message_filters
# ros       
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2


import argparse



# define tensorflow session
parser = argparse.ArgumentParser()

parser.add_argument(
    '--model',
    type=str,
    default = 'resnet50_custom.h5',
    help='provide a model name, save it in snapshots and just pass file name not whole path'
)

parser.add_argument(
    '--res',
    type=int,
    default=600,
    help='provide a resolution to process the images at'
)

args = parser.parse_args()


def get_session():f
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# use this environment flag to change which GPU to use
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())


# adjust this to point to your downloaded/trained model
# models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
#model_path = "/home/autodrive/Desktop/catkin_ws/src/keras-retinanet/snapshots/resnet50_v2.h5"
model_path = "/home/autodrive/Desktop/catkin_ws/src/keras-retinanet/snapshots/" + args.model
model = models.load_model(model_path, backbone_name='resnet50')
model._make_predict_function()

# Camera Detection ROS Node Class
class CameraDetectionROSNode:
    # Initialize class
    def __init__(self):
        self.image = None   # image variable
        self.active = True  # should we be processing images
        self.obstacle_publisher = rospy.Publisherf('/obstacle', ObstacleArray, queue_size = 1)
        self.label_publisher = rospy.Publisher('/labels', String, queue_size = 1)
        self.stamp_publisher = rospy.Publisher('/stamp', Header, queue_size = 1)
        self.pcl_publisher = rospy.Publisher('/campcl', PointCloud2, queue_size = 1)
        
        # CTRL+C signal handler
        signal.signal(signal.SIGINT, self.signalInterruptHandler)
        #self.stamp = 0
        # initialize ros node
        rospy.init_node('camera_detection_node', anonymous=True)
        self.i = 0
        # set camera image topic subscriber
        self.sub_image = rospy.Subscriber('/front_camera/image_raw', Image, self.updateImage, queue_size=1)
        self.sub_cloud = rospy.Subscriber('/lidar_center/velodyne_points', PointCloud2, self.updatePc, queue_size = 1)

        # ros to cv mat converter
        self.bridge = CvBridge()
        # set publishers
        self.detect_pub = rospy.Publisher('/detected', Image, queue_size = 1)
        self.bbox_pub = rospy.Publisher('/bbox', Float32MultiArray, queue_size = 1)
        self.traff_light_pub = rospy.Publisher('/traffic_light',TrafficLightArray, queue_size = 1)
        self.traff_sign_pub = rospy.Publisher('/traffic_sign',TrafficSignArray, queue_size = 1)

        # load label to names mapping for visualization purposes

        if 'coco' in model_path:
            self.labels_to_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat'
                                ,9: 'traffic light', 10: 'fire hydrant', 11: 'stopsign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat'
                               ,16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack'
                               ,25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball'
                               ,33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle'
                               ,40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich'
                               ,49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch'
                               ,58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote'
                               ,66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book'
                               ,74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
        
        else:
            self.labels_to_names = {0: 'greenLight',
                                1 : 'redLight',
                                2 :  'redLightLeft',
                                3 : 'greenLightLeft',
                                4 : 'yellowLight',
                                5 :  'yellowLightLeft',
                                6 : 'car',
                                7 : 'left_turn',
                                8 : 'right_turn',
                                9 : 'Traffic_cones',
                                10 :'parking_sign_handicap',
                                11 : 'do_not_enter',
                                12 :'bicycle',
                                13 :'parking_sign',
                                14 :'stopsign',
                                15 : 'person',
                                16 : 'speed_limit_10',
                                17 : 'speed_limit_15',
                                18 : 'speed_limit_20',
                                19 : 'speed_limit_25',
                                20 : 'speed_limit_5'
            }

        convert_label = {'greenLight' : 5, 'redLight': 1,'greenLightLeft': 4,'yellowLight' : 3,'yellowLightLeft' : 2}
        sign_to_num = {'stopsign' : 1,  'parking_sign' : 2, 'parking_sign_handicap': 3 , 'speed_limit_10': 4, 'speed_limit_15' : 4, 'speed_limit_20' : 4, 'speed_limit_25' : 4, 'speed_limit_5' : 4, 'left_turn' : 5, 'right_turn' :6, 'do_not_enter' : 7}

        # begin processing images
        self.p2 = np.array([[553.144531, 0.000000, 383.442257, 0.000000], [0.000000, 552.951477, 348.178354, 0.000000], [0.000000, 0.000000, 1.000000, 0]])
        self.p2_inv = np.linalg.pinv(self.p2)
        self.processImages()
        self.res = args.res
    # signal interrupt handler, immediate stop of CAN_Controller
    def signalInterruptHandler(self, signum, frame):
        #print("Camera Detection - Exiting ROS Node...")

        # shut down ROS node and stop processess
        rospy.signal_shutdown("CTRL+C Signal Caught.")
        self.active = False

        sys.exit()
    def updatePc(self, msg):
        #self.pcl_publisher = msg.data
        self.pcltosend = msg
    def updateImage(self, image_msg):
        # convert and resize

        image = self.bridge.imgmsg_to_cv2(image_msg, "rgb8")
        cv2.imwrite("image.png",image)
        #image, scale = resize_image(image)
        # save image in class
        #nano = round(image_msg.header.stamp.nsecs/1000000000,1)
        #nano = float(nano)
        self.image = image

    def processImages(self):
        #print("Waiting for Images...")
        while(self.image is None):
            pass

        while(self.active):

            #t1 = time.time()


            self.retinanet(self.image)
            #t2 = time.time()
            #print(t2-t1)
    def getDistance(self, box,width,pWidth,dist,calibration):
        focalPoint = (pWidth * dist)/width
        #print("Focal point",focalPoint)
        focalPoint *= calibration
        pixW = min(abs(box[2]-box[0]),abs(box[3]-box[1]))
        #print(pixW)
        distance = (width*focalPoint)/abs(box[2]-box[0])
        distance = distance/12

        #print("Distance: ",(distance-5)*.3048)#*alpha)

        return (distance-5)*.3048


    def retinanet(self, image):
#        image = image[0:1500,:]
        #self.stamp = rospy.get_rostime()
        #print(self.stamp.secs + round(self.stamp.nsecs/1000000000,1))
        image_copy = image

        # convert cv mat to np array through PIL
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = PIL.Image.fromarray(image)
        image = np.asarray(image.convert('RGB'))[:, :, ::-1].copy()

        # copy to draw on
        draw = image.copy()
        #draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

        # preprocess image for network
        image = preprocess_image(image)
        
        image, scale = resize_image(image,args.res,args.res)
        # process image
        start = time.time()
        #print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        #print("Processing Image...")

        self.pcl_publisher.publish(self.pcltosend)
        boxes, scores, labels = model.predict(np.expand_dims(image, axis=0))
        #print("processing time: ", time.time() - start)

        boxes /= scale

        # visualize detections
        traffic_light_array = TrafficLightArray()
        traffic_sign_array = TrafficSignArray()
        obstacle_array = ObstacleArray()
        array_msg = Float32MultiArray()
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            # scores are sorted so we can break
            if score < 0.5:
                break

            color = label_color(label)


            b = box.astype(int)
            draw_box(draw, b, color=color)

            caption = "{} {:.3f}".format(self.labels_to_names[label], score)
            draw_caption(draw, b, caption)

            name = self.labels_to_names[label]
            
            array_msg.data = list(box)
            self.bbox_pub.publish(array_msg)
           
        try:
            self.detect_pub.publish(self.bridge.cv2_to_imgmsg(draw, "bgr8"))
        except CvBridgeError as e:
            print(e)

if __name__ == "__main__":
    # start camera detection ros node

    CameraDetectionROSNode()
