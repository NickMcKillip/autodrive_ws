#!/usr/bin/env python3

# ============================ Information =============================
# Project:      Texas A&M AutoDrive Challenge - Year 2
# Language:     Python 3.5.2
# ROS Package:  camera_detection
# Repository:   https://github.tamu.edu/kipvasq9/camera_detection
# File Name:    camera_detection.py
# Version:      1.0.0
# Description:  Provide access to Neural Networks through ROS.
# Date:         September 24, 2018
# Author:       Juan Vasquez and Robert Hodge
# Contact:      kipvasq9@tamu.edu

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
ImageFile.LOAD_TRUNCATED_IMAGES = True

# neural network
import tensorflow as tf
import keras
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from path_planner.msg import TrafficLightArray, TrafficLight, TrafficSignArray, TrafficSign, ObstacleArray, Obstacle

# ros
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

# define tensorflow session
def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# use this environment flag to change which GPU to use
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())

# adjust this to point to your downloaded/trained model
# models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
model_path = "/home/autodrive/Desktop/catkin_ws/src/keras-retinanet/snapshots/juan.h5"
model_path = "/home/autodrive/Desktop/catkin_ws/src/keras-retinanet/snapshots/juan.h5"
model = models.load_model(model_path, backbone_name='resnet50')
model._make_predict_function()

# Camera Detection ROS Node Class
class CameraDetectionROSNode:
    # Initialize class
    def __init__(self):
        self.image = None   # image variable
        self.active = True  # should we be processing images
        self.obstacle_publisher = rospy.Publisher('/obstacle', ObstacleArray, queue_size = 1)
        # CTRL+C signal handler
        signal.signal(signal.SIGINT, self.signalInterruptHandler)

        # initialize ros node
        rospy.init_node('camera_detection_node', anonymous=True)

        # set camera image topic subscriber
        self.sub_image = rospy.Subscriber('/front_camera/image_raw', Image, self.updateImage, queue_size=1)

        # ros to cv mat converter
        self.bridge = CvBridge()

        # set publishers
        self.detect_pub = rospy.Publisher('/detected', Image, queue_size = 1)
        self.bbox_pub = rospy.Publisher('/bbox', Float32MultiArray, queue_size = 1)
        self.traff_light_pub = rospy.Publisher('/traffic_light',TrafficLightArray, queue_size = 1)
        self.traff_sign_pub = rospy.Publisher('/traffic_sign',TrafficSignArray, queue_size = 1)

        # load label to names mapping for visualization purposes
        self.labels_to_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat'
                                ,9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat'
                               ,16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack'
                               ,25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball'
                               ,33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle'
                               ,40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich'
                               ,49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch'
                               ,58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote'
                               ,66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book'
                               ,74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

        # begin processing images
        self.p2 = np.array([[553.144531, 0.000000, 383.442257, 0.000000], [0.000000, 552.951477, 348.178354, 0.000000], [0.000000, 0.000000, 1.000000, 0]])
        self.p2_inv = np.linalg.pinv(self.p2)
        self.processImages()

    # signal interrupt handler, immediate stop of CAN_Controller
    def signalInterruptHandler(self, signum, frame):
        #print("Camera Detection - Exiting ROS Node...")

        # shut down ROS node and stop processess
        rospy.signal_shutdown("CTRL+C Signal Caught.")
        self.active = False

        sys.exit()

    def updateImage(self, image_msg):
        # convert and resize
        image = self.bridge.imgmsg_to_cv2(image_msg, "rgb8")
        #image, scale = resize_image(image)
        # save image in class
        self.image = image

    def processImages(self):
        #print("Waiting for Images...")
        while(self.image is None):
            pass

        while(self.active):
            self.retinanet(self.image)

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

    def detect_color(self, image,box):
        img = image

        #img = cv2.imread(imageName)
        #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        #img2 = Image.fromarray(img)
        #img2 = img2.crop((box[0],box[1],box[2],box[3]))
        crop_img = img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]


        img3 = np.array(crop_img)

        #px = img3[10,10]
        ##print(px,px[0])
        #plt.imshow(img3)

        green_score = 0
        red_score = 0

        width = int(box[2] - box[0])
        height = int(box[3] - box[1])


        for x in range(0,width):
            for y in range(0,height):
                px = img3[y,x]
                if( ((px[0]>3) and (px[0] < 98)) and ((px[1]>185) and (px[1] <255)) and ((px[2] > 27) and (px[2] <119))):
                    green_score+= 1
                if( ((px[0]>185) and (px[0] < 255)) and ((px[1]>3) and (px[1] <58)) and ((px[2] > 9) and (px[2] <65))):
                    red_score+= 1

        if(green_score > red_score):
            return 5
        else:
            return 1

    def retinanet(self, image):
        image = image[0:1500,:]

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
        image, scale = resize_image(image)

        # process image
        start = time.time()
        #print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        #print("Processing Image...")
        boxes, scores, labels = model.predict(np.expand_dims(image, axis=0))
        #print("processing time: ", time.time() - start)

        # correct for image scale
        boxes /= scale

        # visualize detections
        traffic_light_array = TrafficLightArray()
        traffic_sign_array = TrafficSignArray()
        obstacle_array = ObstacleArray()
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            # scores are sorted so we can break
            if score < 0.5:
                break

            color = label_color(label)

            b = box.astype(int)
            draw_box(draw, b, color=color)

            caption = "{} {:.3f}".format(self.labels_to_names[label], score)
            draw_caption(draw, b, caption)

            array_msg = Float32MultiArray()

            if(label == 9): #is a traffic light
            #120.5 ft
            #250 inches
                traff_light_msg = TrafficLight()
                what_color = self.detect_color(image_copy,box)
                distance = self.getDistance(box,15.25,84.96498,250,1.25)
                traff_light_msg.type = what_color
                traff_light_msg.x = distance
                traff_light_msg.y = 0

                if what_color == 1:
                    pass
                    #print(box,"red")
                else:
                    pass
                    #print(box,"green")
                traffic_light_array.lights.append(traff_light_msg)

            if(label == 11): #is a stopsign
                traff_sign_msg = TrafficSign()
                #getDistance(self, box,width,pWidth,dist,calibration)
                #distance = self.getDistance(box,30,93.454895,180,1.15)
                distance = self.getDistance(box,30,240.78,160,1.15)

                traff_sign_msg.type = 1
                traff_sign_msg.x = distance
                traff_sign_msg.y = 0
                traffic_sign_array.signs.append(traff_sign_msg)

                #array_msg.data = list(box)
                #self.bbox_pub.publish(array_msg)

            if(label == 2):
                #send info to Nick
                u_min, u_max = box[0]/2.56,box[2]/2.56
                v_min, v_max = box[1]/2.56, box[3]/2.56
                width = u_max - u_min
                if box[2] > 300 and width > 10 and box[0] < 1800:
                    print(box)
                    array_msg.data = list(box)
                    distance = self.getDistance(box,64.4,196,466, 1.13)
                    print("car distance", distance)
                    max_hom = np.array([u_max, v_max, 1])
                    min_hom = np.array([u_min, v_max, 1])
                    max_p = np.dot(self.p2_inv, max_hom)[:3]
                    min_p = np.dot(self.p2_inv, min_hom)[:3]
                    if max_p[2] < 0:
                        max_p[0], max_p[2] = -max_p[0], -max_p[2]
                    if min_p[2] < 0:
                        min_p[0], min_p[2] = -min_p[0], -min_p[2]

                    max_frustum_angle = (np.arctan2(max_p[0], max_p[2])) * -1
                    print("max angle", max_frustum_angle)
                    min_frustum_angle = (np.arctan2(min_p[0], min_p[2])) * -1
                    print("min angle", min_frustum_angle)
                    right = (distance-1) * np.tan(max_frustum_angle)
                    left = (distance-1) * np.tan(min_frustum_angle)

                    obs = Obstacle()
                    obs.x1 = distance
                    obs.x2 = distance + 1.85
                    obs.y1 = right
                    obs.y2 = left

                    obstacle_array.obstacles.append(obs)
                    print("left:", left,"right:", right,"distance", distance)
            #print("I see a " + str(self.labels_to_names[label]) + "(" + str(score) + ")")

        self.obstacle_publisher.publish(obstacle_array)
        self.traff_light_pub.publish(traffic_light_array)
        self.traff_sign_pub.publish(traffic_sign_array)

        plt.figure(figsize=(15, 15))
        plt.axis('off')
        #plt.imshow(draw)
        #plt.show()
        #cv2.imshow("hi", draw)
        #cv2.waitKey(1)

        try:
            self.detect_pub.publish(self.bridge.cv2_to_imgmsg(draw, "bgr8"))
        except CvBridgeError as e:
            pass#print(e)

if __name__ == "__main__":
    # start camera detection ros node
    CameraDetectionROSNode()
