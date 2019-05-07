#!/usr/bin/env python


# 

import rospy
from std_msgs.msg import Float32MultiArray, Float32, Bool
import time

#t = 0,x =0,y=0,timer=0
t=0
def callback(data):
    global x,y,t,stt
    #print('t',t)
    mb=50 # Center_Threshold Hyper Parameter

    if t == 0:
        x = (data[0]+data[2])/2
        y = (data[1]+data[3])/2
    t += 1
    if (x-data[0])**2 + (y-data[1])**2>mb**2:
        stt=1
    
    
    

def Mot_new():
    global x,y,t,timer,stt
    stt=0
    t = 0
    rospy.init_node('MOT_new')
    timer = rospy.get_rostime()
    #print('timer',timer)
    #callback(2)
    rospy.Subscriber("/bbox",Float32MultiArray,callback)
    #callback([100,100])
    dr=rospy.Duration(0.5)
    
    while t<=4 and stt==0:
        if rospy.get_rostime() > timer+dr:
            timer=rospy.get_rostime()
            rospy.Subscriber("/bbox",Float32MultiArray,callback)
            #callback([175,175])
    if stt==0:
        rospy.Publisher('/No_Pedestrian', Bool, queue_size=1).publish(True)
        print('true')
    else:
        print('false')
    
        
        
        
    #rospy.spin()


if __name__=='__main__':
    try:
        Mot_new()
    except rospy.ROSInterruptException:
        pass
