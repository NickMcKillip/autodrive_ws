#!/usr/bin/env python3


import sys
from std_msgs.msg import String
sys.path.append('/home/autodrive/Desktop/catkin_ws/src/utils')
sys.path.append('/home/autodrive/Desktop/catkin_ws/src/Frustum_PointNet')
sys.path.append('/home/autodrive/Desktop/catkin_ws/src/pretrained_models')
import rospy
import numpy as np
import sensor_msgs.point_cloud2
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Header
from geometry_msgs.msg import Point
from torch.autograd import Variable
import sys
from frustum_pointnet import FrustumPointNet
from path_planner.msg import ObstacleArray, Obstacle
import os
import math
import message_filters
from kittiloader import LabelLoader2D3D, calibread, LabelLoader2D3D_sequence
import os
import numpy as np
import datetime

src_dir = "/home/autodrive/Desktop/catkin_ws/src/"
home_dir = src_dir + "keras-retinanet/"
model_dir = src_dir + "pretrained_models/"
#########----------------------------------------------

class DataLoader():

    def __init__(self):
        self.calib = calibread(home_dir + "scripts/006330" + ".txt")
        self.obstacle_publisher = rospy.Publisher('/obstacle', ObstacleArray, queue_size = 1) #pathplanning

        self.point_cloud = None
        self.bbox = None
        self.label_string = ""
        self.frustum_point_clouds = None
        self.frustum_R = None
        self.frustum_angle = None
        self.preds = None
        self.header = None
        self.oldheader = None
        self.lock = False
        '''self.labels_to_names = {0: 'go',
                                1 : 'stop',
                                2 : 'stopLeft',
                                3 : 'goLeft',
                                4 : 'warning',
                                5 :'warningLeft',
                                6 : 'car',
                                7 : 'left_turn',
                                8 : 'right_turn',
                                9 : 'Traffic_cones',
                                10 :'parking_sign_handicap',
                                11 : 'do_not_enter',
                                12 :'bicycle',
                                13 :'parking_sign',
                                14 :'stopsign',
                                15 : 'person'}
         '''
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
 

    def set_point_cloud(self, msg):
        x = []
        for point in sensor_msgs.point_cloud2.read_points(msg, skip_nans=True):
            x.append([point[0], point[1], point[2], 0])
        self.point_cloud = np.array(x).astype(float)
        np.save("pc.npy",self.point_cloud)

    def printobstacle(self,msg):
        self.label_string = msg.data
        pass
        #print("Label: " + msg.data)
    def printstamp(self,msg):
        if self.header:
            self.oldheader = self.header
        self.header = msg
    def set_bbox(self, msg):
        #if np.max(np.array(list(msg.data))) < 1300:
        self.bbox = np.array(list(msg.data))

        ratio = float(2048./800.)
        if self.bbox.any() > 0:
            for index, point in enumerate(self.bbox):
                self.bbox[index] = point/ratio
            self.get_frustum()
        #else:
            #print("locked")



    def removeoutliers(self, bbcloud, k):
        mu = np.mean(bbcloud, axis = 0)
        sigma = np.std(bbcloud, axis = 0)
        return bbcloud[np.all(np.abs((bbcloud - mu)) < k * sigma, axis = 1)]

    def get_frustum(self):

        start = datetime.datetime.now()
        bbox = self.bbox
        point_cloud = self.point_cloud
        objectlist = Float32MultiArray()
        obstacle_array = ObstacleArray()

        calib = self.calib
        p2 = calib["P2"]
        Tr_velo_to_cam_orig = calib["Tr_velo_to_cam"]
        R0_rect_orig = calib["R0_rect"]
        R0_rect = np.eye(4)
        R0_rect[0:3, 0:3] = R0_rect_orig
        Tr_velo_to_cam = np.eye(4)
        Tr_velo_to_cam[0:3, :] = Tr_velo_to_cam_orig

        #print (self.label_string)
        point_cloud = point_cloud[point_cloud[:, 0] > 0, :]
        # remove points that are located too far away from the camera:

        #40 as opposed to 80 from kitti
        point_cloud = point_cloud[point_cloud[:, 0] < 40, :]
        point_cloud_xyz = point_cloud[:, 0:3]
        point_cloud_xyz_hom = np.ones((point_cloud.shape[0], 4))
        point_cloud_xyz_hom[:, 0:3] = point_cloud[:, 0:3] # (point_cloud_xyz_hom has shape (num_points, 4))

        # project the points onto the image plane (homogeneous coords):
        img_points_hom = np.dot(p2, np.dot(R0_rect, np.dot(Tr_velo_to_cam, point_cloud_xyz_hom.T))).T # (point_cloud_xyz_hom.T has shape (4, num_points))
        # normalize:
        img_points = np.zeros((img_points_hom.shape[0], 2))
        img_points[:, 0] = img_points_hom[:, 0]/img_points_hom[:, 2]
        img_points[:, 1] = img_points_hom[:, 1]/img_points_hom[:, 2]

        # transform the points into (rectified) camera coordinates:
        point_cloud_xyz_camera_hom = np.dot(R0_rect, np.dot(Tr_velo_to_cam, point_cloud_xyz_hom.T)).T # (point_cloud_xyz_hom.T has shape (4, num_points))
        # normalize:
        point_cloud_xyz_camera = np.zeros((point_cloud_xyz_camera_hom.shape[0], 3))
        point_cloud_xyz_camera[:, 0] = point_cloud_xyz_camera_hom[:, 0]/point_cloud_xyz_camera_hom[:, 3]
        point_cloud_xyz_camera[:, 1] = point_cloud_xyz_camera_hom[:, 1]/point_cloud_xyz_camera_hom[:, 3]
        point_cloud_xyz_camera[:, 2] = point_cloud_xyz_camera_hom[:, 2]/point_cloud_xyz_camera_hom[:, 3]

        point_cloud_camera = point_cloud
        point_cloud_camera[:, 0:3] = point_cloud_xyz_camera

        u_min = self.bbox[0] # (left)
        u_max = self.bbox[2] # (right)
        v_min = self.bbox[1] # (top)
        v_max = self.bbox[3] # (bottom)
        # expand the 2D bbox slightly:
        u_min_expanded = u_min
        u_max_expanded = u_max
        v_min_expanded = v_min
        v_max_expanded = v_max

        row_mask = np.logical_and(
	            np.logical_and(img_points[:, 0] >= u_min_expanded,
			           img_points[:, 0] <= u_max_expanded),
	            np.logical_and(img_points[:, 1] >= v_min_expanded,
			           img_points[:, 1] <= v_max_expanded))

        frustum_point_cloud_xyz = point_cloud_xyz[row_mask, :]

        #ind = np.argmin((frustum_point_cloud_xyz[:,2]))
        #minpoint = frustum_point_cloud_xyz[ind,:]
        #print (minpoint,"minpoint")
        mean = np.mean(self.removeoutliers(frustum_point_cloud_xyz,1), axis = 0)
        distance = math.sqrt(mean[0]**2 + mean[1]**2 + mean[2]**2)
        #objectlist.data = [float(self.label_string), *mean, distance]
        obs = Obstacle()
        
        x = mean[2]
        y = -mean[0]
        z = mean[1]
        obs.x = x 
        obs.y = y
        obs.z = z 
        obs.type = int(self.label_string)
        obstacle_array.obstacles.append(obs)
        self.obstacle_publisher.publish(obstacle_array)
        print (self.labels_to_names[int(self.label_string)] + " x: " + str(mean[0]) + " y: " + str(mean[1]) + " z: " + str(mean[2]))
        #label = float(self.label_string)

        print ("Distance: " + str(distance))
        end = datetime.datetime.now()
      #  self.camheader = self.oldheader
        #print("time: " + str(end-start))


if __name__ == "__main__":

    data = DataLoader()
    print("lidar_main_here")

    # setup signal interrupt handler
    #signal.signal(signal.SIGINT, signalInterruptHandler)

    # variables for network
    #bridge = CvBridge()
    # setup ros semantics
    rospy.Subscriber('/campcl', PointCloud2, data.set_point_cloud)
    rospy.Subscriber('labels', String, data.printobstacle)
    rospy.Subscriber('stamp', Header, data.printstamp)

    rospy.Subscriber('/bbox', Float32MultiArray, data.set_bbox)
    rospy.init_node('lidar_node', anonymous=True)
    rospy.spin()
