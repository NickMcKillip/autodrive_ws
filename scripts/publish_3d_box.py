#!/usr/bin/env python3

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
import rospy
import math
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Point

if __name__ == "__main__":
	rospy.init_node('visualize_3d_box_node')
	publisher = rospy.Publisher('3d_box', MarkerArray, queue_size=1)
	rospy.Subscriber("autodrive_sim/output/waypoints", Float32MultiArray, waypoint_cb)
	rospy.Subscriber("nonactive_autodrive_sim/output/waypoints", Float32MultiArray, waypoint_cb2)
	rospy.spin()
