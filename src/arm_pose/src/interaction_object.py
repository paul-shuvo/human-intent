#!/usr/bin/env python
import rospy
from arm_pose.msg import Floats
# import ros_np_multiarray as rnm
import numpy as np
import json
# from rospy.numpy_msg import numpy_msg
import message_filters
from std_msgs.msg import String
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from geometry_msgs.msg import PoseStamped

from sympy import Point3D, Plane, symbols, N
from sympy.geometry import Line3D, Segment
# import cv2
# import os
# from roslib import message
# import numpy as np
#listener

# class Paul:
#     def __init__(self):
#         self.job = 'CS Grad Student'
#         self.hobbies = ['Playing Flute', 'Dancing Bachata']




class ObjectInteraction:
    def __init__(self):  
        """
        Initializes the class and starts the subscribers.

        Args:
            canvas_pts (`ndarray`): Three pixel locations (co-planar in 3D space) on an image that describe a surface.
        """
        # self.K is the camera matrix retrieved from /kinect2/qhd/camera_info 
        self.K = np.array([540.68603515625, 0.0, 479.75, 0.0, 540.68603515625, 269.75, 0.0, 0.0, 1.0]).reshape((3,3), order='C')
        # self.canvas_pts = canvas_pts
        # self.canvas_pts_3D = np.zeros((len(self.canvas_pts), 3))

        # initialize arm points in 2D and 3D
        self.arm_points = np.zeros((2, 2), dtype=np.int16).tolist()
        self.arm_points_3D = np.random.random_sample((2, 3))
        self.detected_object_plane = {}
        self.t = symbols('t')

        rospy.init_node('object_selection_node', anonymous=False)
        self.pub = rospy.Publisher('/object_selected', PoseStamped, queue_size=10)
        self.msg = PoseStamped()
        self.frame_id = 'kinect2_rgb_optical_frame'
        # forearm_pose_sub = message_filters.Subscriber('/kinect2/qhd/camera_info', CameraInfo)
        forearm_pose_sub = message_filters.Subscriber('/forearm_pose', Floats)
        pointcloud_sub = message_filters.Subscriber('/kinect2/qhd/points', PointCloud2)
        detected_object_sub = message_filters.Subscriber('/detected_object', String)      

        ts = message_filters.ApproximateTimeSynchronizer([forearm_pose_sub, pointcloud_sub, detected_object_sub], 10, 1, allow_headerless=True) # Changed code
        ts.registerCallback(self.callback)
        # spin
        rospy.spin()
    
    def update_points(self, forearm_pose, pointcloud, detected_object):
        # choose left arm for now
        arm_loc_np = np.asarray(forearm_pose.data, dtype=np.int16)
        left_arm_joint_pts = [0, 2]
        right_arm_joint_pts = [1, 3]
        self.arm_points = arm_loc_np.reshape((arm_loc_np.shape[0]//2, -1), order='C')[left_arm_joint_pts].tolist()

        
        pre_arm_points_3D = self.arm_points_3D
        count = 0
        for dt in pc2.read_points(pointcloud, field_names={'x','y','z'}, skip_nans=False, uvs=self.arm_points):
            # check if it's not NaN
            if dt[0] == dt[0]:
                # print(f'x: {dt[0]}, y: {dt[1]}, z: {dt[2]}')
                self.arm_points_3D[count, :] = dt
            else:
                self.arm_points_3D = pre_arm_points_3D
                break
            count += 1
        # print(self.arm_points_3D)
        pre_detected_object_plane = self.detected_object_plane
        t_val = 0.75
        # if erros using the json.load() check the following link
        # https://stackoverflow.com/questions/11174024/attributeerrorstr-object-has-no-attribute-read
        for object_name, bounding_box in json.loads(detected_object.data).items():
            if bounding_box is None:
                continue
            # try:
            # edge_pts = zip(pre_detected_object_dict[object_name], [*pre_detected_object_dict[object_name][1:], pre_detected_object_dict[object_name][0]])
            # print(bounding_box)
            tl, bl, br, tr = np.array(bounding_box)
            center, right_c, top_c = (tl + br) // 2, (tr + br) // 2, (tl + tr) // 2 
            # print(center)
            # print(right_c)
            # segments = [Segment(tuple(pts[0]), tuple(pts[1])) for pts in edge_pts]
            arb_x = [int(i) for i in list(Segment(tuple(center), tuple(right_c)).arbitrary_point(self.t).subs(self.t, t_val).evalf())]
            arb_y = [int(i) for i in list(Segment(tuple(center), tuple(top_c)).arbitrary_point(self.t).subs(self.t, t_val).evalf())]
            
            nan_flag = False
            pts_3d = []
            for dt in pc2.read_points(pointcloud, field_names={'x','y','z'}, skip_nans=False, uvs=[arb_x, arb_y]):
                # check if it's not NaN
                if dt[0] == dt[0]:
                    # print(f'x: {dt[0]}, y: {dt[1]}, z: {dt[2]}')
                    pts_3d.append(tuple(dt))
                else:
                    nan_flag = True
                    break
            
            if not nan_flag:
                self.detected_object_plane[object_name] = Plane(*pts_3d)
            # except TypeError as e:
            #     rospy.loginfo(e)
            #     continue
        print('end')

    def callback(self, forearm_pose, pointcloud, detected_object):
        self.update_points(forearm_pose, pointcloud, detected_object)
        print(self.detected_object_plane)
        # try:
        #     left_arm_line_3D = Line3D(Point3D(*self.arm_points_3D[0, :]), Point3D(*self.arm_points_3D[1, :]))
        #     canvas_plane = Plane(Point3D(*self.canvas_pts_3D[0, :]), Point3D(*self.canvas_pts_3D[1, :]), Point3D(*self.canvas_pts_3D[2, :]))
        #     canvas_plane_point = canvas_plane.intersection(left_arm_line_3D)[0].evalf()
            
        #     self.msg.header.stamp = rospy.Time.now()
        #     self.msg.header.frame_id = 'kinect2_rgb_optical_frame'
        #     self.msg.pose.position.x = canvas_plane_point.x
        #     self.msg.pose.position.y = canvas_plane_point.y
        #     self.msg.pose.position.z = canvas_plane_point.z
        #     # Make sure the quaternion is valid and normalized
        #     self.msg.pose.orientation.x = 0.0
        #     self.msg.pose.orientation.y = 0.0
        #     self.msg.pose.orientation.z = 0.0
        #     self.msg.pose.orientation.w = 1.0
        #     self.frame_id += 1
        #     # print(canvas_plane_point)
        #     # print(canvas_plane_point.y)
        # except ValueError as error:
        #     rospy.loginfo(error)

        # self.pub.publish(self.msg)
        # print('msg published')



def read_depth(width, height, data) :
    # read function
    if (height >= data.height) or (width >= data.width) :
        return -1
    data_out = pc2.read_points(data, field_names=None, skip_nans=False, uvs=[[width, height]])
    int_data = next(data_out)
    rospy.loginfo("int_data " + str(int_data))
    return int_data


if __name__ == '__main__':
    ObjectInteraction()
    # try:
    #     listen()
    # except rospy.ROSInterruptException as error:
    #     rospy.loginfo(error)