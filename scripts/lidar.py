#!/usr/bin/env python3


#def func(custom_msg):

#src_dir = "/home/autodrive/Desktop/catkin_ws/src/"
#home_dir = src_dir + "Frustum-PointNet/"
#model_dir = src_dir + "pretrained_models/"


import sys
#src_dir = "/home/autodrive/Desktop/catkin_ws/src/"'
sys.path.append('/home/autodrive/Desktop/catkin_ws/src/utils')
sys.path.append('/home/autodrive/Desktop/catkin_ws/src/Frustum_PointNet')
sys.path.append('/home/autodrive/Desktop/catkin_ws/src/pretrained_models')
import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2
from std_msgs.msg import Float32MultiArray
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
from frustum_pointnet import FrustumPointNet

import os


from datasets import EvalDatasetFrustumPointNet, wrapToPi, getBinCenter
from kittiloader import LabelLoader2D3D, calibread, LabelLoader2D3D_sequence
import torch.utils.data
import os
import pickle
import numpy as np
import datetime

src_dir = "/home/autodrive/Desktop/catkin_ws/src/"
home_dir = src_dir + "keras-retinanet/"
model_dir = src_dir + "pretrained_models/"


class DataLoader():

    def __init__(self):

        with open(home_dir + "scripts/val_img_ids.pkl", "rb") as file: # (needed for python3)
            img_ids = pickle.load(file)

        with open(home_dir + "scripts/kitti_train_mean_car_size.pkl", "rb") as file: # (needed for python3)
            self.mean_car_size = pickle.load(file)

        with open(home_dir + "scripts/kitti_centered_frustum_mean_xyz.pkl", "rb") as file: # (needed for python3)
            self.centered_frustum_mean_xyz = pickle.load(file)
            self.centered_frustum_mean_xyz = self.centered_frustum_mean_xyz.astype(np.float32)

        self.network = FrustumPointNet("Frustum-PointNet_eval_test", project_dir=home_dir)
        self.network.load_state_dict(torch.load(model_dir + "model_37_2_epoch_400.pth", map_location='cpu'))
        self.NH = self.network.BboxNet_network.NH

        self.calib = calibread(home_dir + "scripts/006330" + ".txt")

        self.point_cloud = None

        self.bbox = None

        self.frustum_point_clouds = None
        self.frustum_R = None
        self.frustum_angle = None
        self.preds = None




    def set_point_cloud(self, msg):
        x = []
        for point in sensor_msgs.point_cloud2.read_points(msg, skip_nans=True):
            x.append([point[0], point[1], point[2], 0])

        self.point_cloud = np.array(x).astype(float)


    def set_bbox(self, msg):
        self.bbox = np.array(list(msg.data))

        print(self.bbox)

        self.get_frustum()

    def get_frustum(self):
        bbox = self.bbox
        point_cloud = self.point_cloud

        # remove points that are located behind the camera:
        point_cloud = point_cloud[point_cloud[:, 0] > 0, :]
        # remove points that are located too far away from the camera:
        point_cloud = point_cloud[point_cloud[:, 0] < 80, :]

        calib = self.calib
        P2 = calib["P2"]
        Tr_velo_to_cam_orig = calib["Tr_velo_to_cam"]
        R0_rect_orig = calib["R0_rect"]
        #
        R0_rect = np.eye(4)
        R0_rect[0:3, 0:3] = R0_rect_orig
        #
        Tr_velo_to_cam = np.eye(4)
        Tr_velo_to_cam[0:3, :] = Tr_velo_to_cam_orig

        point_cloud_xyz = point_cloud[:, 0:3]
        point_cloud_xyz_hom = np.ones((point_cloud.shape[0], 4))
        point_cloud_xyz_hom[:, 0:3] = point_cloud[:, 0:3] # (point_cloud_xyz_hom has shape (num_points, 4))

        # project the points onto the image plane (homogeneous coords):
        img_points_hom = np.dot(P2, np.dot(R0_rect, np.dot(Tr_velo_to_cam, point_cloud_xyz_hom.T))).T # (point_cloud_xyz_hom.T has shape (4, num_points))
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
        #label_2D = example["label_2D"]
        #label_3D = example["label_3D"]

        #bbox = label_2D["poly"]

        # img = cv2.imread(self.img_dir + img_id + ".png", -1)
        # img_with_bboxes = draw_2d_polys(img, [label_2D])
        # cv2.imwrite("test.png", img_with_bboxes)

        ########################################################################
        # frustum:
        ########################################################################

        u_min = self.bbox[0] # (left)
        u_max = self.bbox[2] # (right)
        v_min = self.bbox[1] # (top)
        v_max = self.bbox[3] # (bottom)
        #set_trace()
        #u_min = bbox[0, 0] # (left)
        #u_max = bbox[1, 0] # (rigth)
        #v_min = bbox[0, 1] # (top)
        #v_max = bbox[2, 1] # (bottom)
        # expand the 2D bbox slightly:
        u_min_expanded = u_min #- (u_max-u_min)*0.05
        u_max_expanded = u_max #+ (u_max-u_min)*0.05
        v_min_expanded = v_min #- (v_max-v_min)*0.05
        v_max_expanded = v_max #+ (v_max-v_min)*0.05
        input_2Dbbox = np.array([u_min_expanded, u_max_expanded, v_min_expanded, v_max_expanded])

        row_mask = np.logical_and(
                    np.logical_and(img_points[:, 0] >= u_min_expanded,
                                   img_points[:, 0] <= u_max_expanded),
                    np.logical_and(img_points[:, 1] >= v_min_expanded,
                                   img_points[:, 1] <= v_max_expanded))

        frustum_point_cloud_xyz = point_cloud_xyz[row_mask, :] # (needed only for visualization)
        frustum_point_cloud = point_cloud[row_mask, :]
        frustum_point_cloud_xyz_camera = point_cloud_xyz_camera[row_mask, :]
        frustum_point_cloud_camera = point_cloud_camera[row_mask, :]

        #if frustum_point_cloud.shape[0] == 0:
         #   return self.__getitem_-(0)
        # randomly sample 1024 points in the frustum point cloud:
        if frustum_point_cloud.shape[0] < 1024:
            row_idx = np.random.choice(frustum_point_cloud.shape[0], 1024, replace=True)
        else:
            row_idx = np.random.choice(frustum_point_cloud.shape[0], 1024, replace=False)
        frustum_point_cloud_xyz = frustum_point_cloud_xyz[row_idx, :]
        frustum_point_cloud = frustum_point_cloud[row_idx, :]
        frustum_point_cloud_xyz_camera = frustum_point_cloud_xyz_camera[row_idx, :]
        frustum_point_cloud_camera = frustum_point_cloud_camera[row_idx, :]
        ########################################################################
        # InstanceSeg ground truth:
        ########################################################################


        ########################################################################
        # normalize frustum point cloud:
        ########################################################################
        # get the 2dbbox center img point in hom. coords:
        u_center = u_min_expanded + (u_max_expanded - u_min_expanded)/2.0
        v_center = v_min_expanded + (v_max_expanded - v_min_expanded)/2.0
        center_img_point_hom = np.array([u_center, v_center, 1])

        # (more than one 3D point is projected onto the center image point, i.e,
        # the linear system of equations is under-determined and has inf number
        # of solutions. By using the pseudo-inverse, we obtain the least-norm sol)

        # get a point (the least-norm sol.) that projects onto the center image point, in hom. coords:
        P2_pseudo_inverse = np.linalg.pinv(P2) # (shape: (4, 3)) (P2 has shape (3, 4))
        point_hom = np.dot(P2_pseudo_inverse, center_img_point_hom)

        # hom --> normal coords:
        point = np.array(([point_hom[0]/point_hom[3], point_hom[1]/point_hom[3], point_hom[2]/point_hom[3]]))

        # if the point is behind the camera, switch to the mirror point in front of the camera:
        if point[2] < 0:
            point[0] = -point[0]
            point[2] = -point[2]

        # compute the angle of the point in the x-z plane: ((rectified) camera coords)
        self.frustum_angle = np.arctan2(point[0], point[2]) # (np.arctan2(x, z)) # (frustum_angle = 0: frustum is centered)

        # rotation_matrix to rotate points frustum_angle around the y axis (counter-clockwise):
        self.frustum_R = np.asarray([[np.cos(self.frustum_angle), 0, -np.sin(self.frustum_angle)],
                           [0, 1, 0],
                           [np.sin(self.frustum_angle), 0, np.cos(self.frustum_angle)]],
                           dtype='float32')

        # rotate the frustum point cloud to center it:
        centered_frustum_point_cloud_xyz_camera = np.dot(self.frustum_R, frustum_point_cloud_xyz_camera.T).T

        # subtract the centered frustum train xyz mean:
        centered_frustum_point_cloud_xyz_camera -= self.centered_frustum_mean_xyz

        centered_frustum_point_cloud_camera = frustum_point_cloud_camera
        centered_frustum_point_cloud_camera[:, 0:3] = centered_frustum_point_cloud_xyz_camera



        self.frustum_point_clouds = torch.from_numpy(centered_frustum_point_cloud_camera) # (shape: (1024, 4))
        self.frustum_point_clouds = self.frustum_point_clouds.permute(1,0)
        self.pred()



    def pred(self):
        self.preds = self.network(torch.unsqueeze(self.frustum_point_clouds,dim=0).float())

        self.preds_to_3dbb()

    def preds_to_3dbb(self):
        i = 0
        centered_frustum_mean_xyz = np.array(self.centered_frustum_mean_xyz)
        mean_car_size = np.array(self.mean_car_size)
        outputs_InstanceSeg = self.preds[0] # (shape: (batch_size, num_points, 2))
        outputs_TNet = self.preds[1] # (shape: (batch_size, 3))
        outputs_BboxNet = self.preds[2] # (shape: (batch_size, 3 + 3 + 2*NH))
        seg_point_clouds_mean = self.preds[3] # (shape: (batch_size, 3))
        dont_care_mask = self.preds[4] # (shape: (batch_size, ))

        pred_InstanceSeg = outputs_InstanceSeg.data.cpu().numpy() # (shape: (num_points, 2))
        frustum_point_cloud = self.frustum_point_clouds.data.cpu().numpy() # (shape: (num_points, 4))
        frustum_point_cloud = np.transpose(frustum_point_cloud,(1,0))
        seg_point_cloud_mean = seg_point_clouds_mean.data.cpu().numpy() # (shape: (3, ))
        #img_id = img_ids[i]
        input_2Dbbox = self.bbox # (shape: (4, ))
        frustum_R = self.frustum_R # (shape: (3, 3))
        frustum_angle = self.frustum_angle
        #score_2d = scores_2d[i]

        unshifted_frustum_point_cloud_xyz = frustum_point_cloud[:, 0:3] + centered_frustum_mean_xyz
        unshifted_frustum_point_cloud_xyz = np.array(unshifted_frustum_point_cloud_xyz)
        decentered_frustum_point_cloud_xyz = np.dot(np.linalg.inv(frustum_R), unshifted_frustum_point_cloud_xyz.T).T
        frustum_point_cloud[:, 0:3] = decentered_frustum_point_cloud_xyz

        #print(pred_InstanceSeg.shape)
        pred_InstanceSeg = np.squeeze(pred_InstanceSeg)
        row_mask = pred_InstanceSeg[:, 1] > pred_InstanceSeg[:, 0]
       # row_mask = np.squeeze(row_mask)
       # print(row_mask.shape)

        #print(frustum_point_cloud.shape)
        pred_seg_point_cloud = frustum_point_cloud[row_mask, :]
        
        
        pred_center_TNet = np.dot(np.linalg.inv(frustum_R), np.squeeze(outputs_TNet[i].data.cpu().numpy()) + np.squeeze(np.array(centered_frustum_mean_xyz)) +np.squeeze( seg_point_cloud_mean)) # (shape: (3, )) # NOTE!
        centroid = seg_point_cloud_mean

        pred_center_BboxNet = np.dot(np.linalg.inv(frustum_R), np.squeeze(outputs_BboxNet[i][0:3].data.cpu().numpy()) + np.squeeze( centered_frustum_mean_xyz) +np.squeeze( seg_point_cloud_mean) + np.squeeze(outputs_TNet[i].data.cpu().numpy())) # (shape: (3, )) # NOTE!

        pred_h = outputs_BboxNet[i][3].data.cpu().numpy() + mean_car_size[0]
        pred_w = outputs_BboxNet[i][4].data.cpu().numpy() + mean_car_size[1]
        pred_l = outputs_BboxNet[i][5].data.cpu().numpy() + mean_car_size[2]

        pred_bin_scores = outputs_BboxNet[i][6:(6+4)].data.cpu().numpy() # (shape (NH=8, ))
        pred_residuals = outputs_BboxNet[i][(6+4):].data.cpu().numpy() # (shape (NH=8, ))
        pred_bin_number = np.argmax(pred_bin_scores)
        pred_bin_center = getBinCenter(pred_bin_number, NH=self.NH)
        pred_residual = pred_residuals[pred_bin_number]
        pred_centered_r_y = pred_bin_center + pred_residual
        pred_r_y = wrapToPi(pred_centered_r_y + frustum_angle) # NOTE!

        #pred_r_y = pred_r_y.data.numpy()
        #input_2Dbbox = input_2Dbbox.data.numpy()

        bbox_dict = {}
        # # # # uncomment this if you want to visualize the frustum or the segmentation:
        # bbox_dict["frustum_point_cloud"] = frustum_point_cloud
        # bbox_dict["pred_seg_point_cloud"] = pred_seg_point_cloud
        # # # #
        bbox_dict["pred_center_TNet"] = pred_center_TNet
        bbox_dict["pred_center_BboxNet"] = pred_center_BboxNet
        bbox_dict["centroid"] = centroid
        bbox_dict["pred_h"] = pred_h
        bbox_dict["pred_w"] = pred_w
        bbox_dict["pred_l"] = pred_l
        bbox_dict["pred_r_y"] = pred_r_y
        bbox_dict["input_2Dbbox"] = input_2Dbbox

        print(bbox_dict)



        return bbox_dict



if __name__ == "__main__":

    data = DataLoader()


    # setup signal interrupt handler
    #signal.signal(signal.SIGINT, signalInterruptHandler)

    # variables for network
    #bridge = CvBridge()
    # setup ros semantics
    rospy.Subscriber('lidar_center/velodyne_points', PointCloud2, data.set_point_cloud)
    rospy.Subscriber('/bbox', Float32MultiArray, data.set_bbox)
    rospy.init_node('lidar_node', anonymous=True)
    rospy.spin()
