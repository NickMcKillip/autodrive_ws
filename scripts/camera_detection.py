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
from std_msgs.msg import Float32MultiArray
ImageFile.LOAD_TRUNCATED_IMAGES = True

# neural network
import tensorflow as tf
import keras
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from path_planner.msg import TrafficLightArray, TrafficLight

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
model = models.load_model(model_path, backbone_name='resnet50')
model._make_predict_function()

# load label to names mapping for visualization purposes
labels_to_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

detect_pub = rospy.Publisher('/detected', Image, queue_size = 1)

bbox_pub = rospy.Publisher('/bbox', Float32MultiArray, queue_size = 1)

traff_light_pub = rospy.Publisher('/traffic_light',TrafficLightArray, queue_size = 1)

# signal interrupt handler, immediate stop of CAN_Controller
def signalInterruptHandler(signum, frame):
    print("Camera Detection - Exiting ROS Node...")

    # shut down ROS node
    rospy.signal_shutdown("CTRL+C Signal Caught.")

    sys.exit()

def updateImage(image_msg):
    bridge = CvBridge()

    image = bridge.imgmsg_to_cv2(image_msg, "bgr8")
    image, scale = resize_image(image)

    retinanet(image)

def getDistance(box,width):
    print(box)
    #print("x1-x2 = ", abs(box[0]-box[2]))
    #print("y1-y2 = ", abs(box[1]-box[3]))
    #area = abs(box[0]-box[2])*abs(box[1]-box[3])
    #print("Area = ",area)

    focalPoint = (93.454895 * 160)/width
    focalPoint *= 1.15
    distance = (width*focalPoint)/abs(box[0]-box[2])
    distance = distance/12
    #alpha = float(box[2]-box[0])
    #alpha = alpha/100
    #print("Alpha: ",alpha)

    print("Distance: ",(distance-5)*.3048)#*alpha)

    return (distance-5)*.3048

def detect_color(image,box):
    img = image

    #img = cv2.imread(imageName)
    #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    #img2 = Image.fromarray(img)
    #img2 = img2.crop((box[0],box[1],box[2],box[3]))
    crop_img = img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]


    img3 = np.array(crop_img)

    #px = img3[10,10]
    #print(px,px[0])
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

    if(red_score > green_score):
        return 1
    else:
        return 5

def retinanet(image):
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
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Processing Image...")
    boxes, scores, labels = model.predict(np.expand_dims(image, axis=0))
    print("processing time: ", time.time() - start)

    # correct for image scale
    boxes /= scale

    # visualize detections
    """traffic_light_array = TrafficLightArray()"""
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < 0.5:
            break

        color = label_color(label)

        b = box.astype(int)
        draw_box(draw, b, color=color)

        caption = "{} {:.3f}".format(labels_to_names[label], score)
        draw_caption(draw, b, caption)

        array_msg = Float32MultiArray()

        if(label == 9): #is a traffic light
        #120.5 ft
        #250 inches
            """traff_light_msg = TrafficLight()"""
            what_color = detect_color(image_copy,box)
            """traff_light_msg.type = what_color
            traff_light_msg.x = distance
            traff_light_msg.y = 0"""

            if what_color == 1:
                print(box,"red")
            else:
                print(box,"green")
            """traffic_light_array.lights.append(traff_light_msg)"""

        if(label == 11): #is a stopsign
            distance = getDistance(box,30)
            array_msg.data = list(box)
            bbox_pub.publish(array_msg)

        if(label == 2):
            #send info to Nick
            array_msg.data = list(box)
            bbox_pub.publish(array_msg)

        print("I see a " + str(labels_to_names[label]) + "(" + str(score) + ")")

    """traff_light_pub.publish(traffic_light_array)"""

    plt.figure(figsize=(15, 15))
    plt.axis('off')
    #plt.imshow(draw)
    #plt.show()
    #cv2.imshow("hi", draw)
    #cv2.waitKey(1)
    bridge = CvBridge()
    try:
        detect_pub.publish(bridge.cv2_to_imgmsg(draw, "bgr8"))
    except CvBridgeError as e:
        print(e)

if __name__ == "__main__":
    # setup signal interrupt handler
    signal.signal(signal.SIGINT, signalInterruptHandler)

    # variables for network
    bridge = CvBridge()

    # setup ros semantics
    rospy.init_node('camera_detection_node', anonymous=True)

    sub_image = rospy.Subscriber('/front_camera/image_raw', Image, updateImage, queue_size=1, buff_size=26000000)

    rospy.spin()

    #while not rospy.is_shutdown():
    #    #image = read_image_bgr("/home/perception/Desktop/Bobby/keras-retinanet/examples/juan.JPG")
    #       #image = cv2.imread("/home/perception/Desktop/Bobby/keras-retinanet/examples/juan.JPG")
    #    retinanet()











#
# # Camera Detection ROS Node Class
# class CameraDetectionROSNode:
#     # Initialize class
#     def __init__(self):
#         self.image = None   # image variable
#         self.active = True  # should we be processing images
#
#         # CTRL+C signal handler
#         signal.signal(signal.SIGINT, self.signalInterruptHandler)
#
#         # initialize ros node
#         rospy.init_node('camera_detection_node', anonymous=True)
#
#         # set camera image topic subscriber
#         self.sub_image = rospy.Subscriber('/front_camera/image_raw', Image,
#                                            self.imageCallback, queue_size=1)
#
#         # begin processing images
#         self.processImages()
#
#
#     # ROS subscriber callback
#     def imageCallback(self, image_msg):
#         bridge = CvBridge()
#
#         # check if valid camera data
#         if image_msg.data is None:
#             raise RunTimeError("Null image data from read message.")
#
#         # convert to cv mat
#         image = bridge.imgmsg_to_cv2(image_msg, "bgr8")
#
#         # decode failure
#         if image is None:
#             raise RunTimeError("Decoded image is null.")
#
#         # perform detection and save image to output;
#         self.image = self.retinanet(image)
#
#     # Resize image from ROS
#     def resizeImage(self, img, target_size):
#         try:
#             # load parameters
#             target_size = float(target_size)
#             height = img.shape[0]
#             width = img.shape[1]
#
#             # compute scaling factor, apply resizing
#             if height > width:
#                 scale_factor = target_size / height
#                 img = cv2.resize(img, (int(width * scale_factor), int(height * scale_factor)))
#             else:
#                 scale_factor = target_size / width
#                 img = cv2.resize(img, (int(width * scale_factor), int(height * scale_factor)))
#         except:
#             pass # empty image
#
#         return img
#
#     # Show OpenCV image
#     def showImage(self, name, image):
#         try:
#             cv2.imshow(name, image)
#             cv2.waitKey(0) # change to 0 for manual iteration through images (press key to iterate)
#         except:
#             pass
#
#     # Process images through pipeline
#     def processImages(self):
#         while self.active:
#             resized_image = self.resizeImage(self.image, 768)
#             self.showImage('current image', resized_image)
#
#     # signal interrupt handler, immediate stop of CAN_Controller
#     def signalInterruptHandler(self, signum, frame):
#         # print interrupt
#         print("Camera Detection - Exiting ROS Node...")
#
#         # immediately reset controlMode/checking variables
#         self.active = False
#
#         # shut down ROS node
#         rospy.signal_shutdown("CTRL+C Signal Caught.")
#
