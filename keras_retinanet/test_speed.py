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
from std_msgs.msg import String
# neural network
import tensorflow as tf
import keras
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from path_planner.msg import TrafficLightArray, TrafficLight, TrafficSignArray, TrafficSign, ObstacleArray, Obstacle


model_path = "/home/autodrive/Desktop/catkin_ws/src/keras-retinanet/snapshots/juan.h5"
model_path = "/home/autodrive/Desktop/catkin_ws/src/keras-retinanet/snapshots/juan.h5"
model = models.load_model(model_path, backbone_name='resnet50')


