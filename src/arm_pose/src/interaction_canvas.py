#!/usr/bin/env python
import rospy
from arm_pose.msg import Floats
# import ros_np_multiarray as rnm
import numpy as np
# from rospy.numpy_msg import numpy_msg
import message_filters
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from geometry_msgs.msg import PoseStamped

from sympy import Point3D, Plane
from sympy.geometry import Line3D
# import cv2
# import os
# from roslib import message
# import numpy as np
#listener

# class Paul:
#     def __init__(self):
#         self.job = 'CS Grad Student'
#         self.hobbies = ['Playing Flute', 'Dancing Bachata']




class CanvasInteraction:
    def __init__(self, canvas_pts):  
        """
        Initializes the class and starts the subscribers.

        Args:
            canvas_pts (`ndarray`): Three pixel locations (co-planar in 3D space) on an image that describe a surface.
        """
        # self.K is the camera matrix retrieved from /kinect2/qhd/camera_info 
        self.K = np.array([540.68603515625, 0.0, 479.75, 0.0, 540.68603515625, 269.75, 0.0, 0.0, 1.0]).reshape((3,3), order='C')
        self.canvas_pts = canvas_pts
        self.canvas_pts_3D = np.zeros((len(self.canvas_pts), 3))
        self.arm_points = np.zeros((2, 2), dtype=np.int16).tolist()
        self.arm_points_3D = np.random.random_sample((2, 3))

        rospy.init_node('projection_node', anonymous=False)
        self.pub = rospy.Publisher('/point_on_canvas', PoseStamped, queue_size=10)
        self.msg = PoseStamped()
        self.frame_id = 'kinect2_rgb_optical_frame'
        # forearm_pose_sub = message_filters.Subscriber('/kinect2/qhd/camera_info', CameraInfo)
        forearm_pose_sub = message_filters.Subscriber('/forearm_pose', Floats)
        pointcloud_sub = message_filters.Subscriber('/kinect2/qhd/points', PointCloud2)      
        ts = message_filters.ApproximateTimeSynchronizer([forearm_pose_sub, pointcloud_sub], 10, 1, allow_headerless=True) # Changed code
        ts.registerCallback(self.callback)
        # spin
        rospy.spin()
    
    def update_points(self, forearm_pose, pointcloud):
        # choose left arm for now
        arm_loc_np = np.asarray(forearm_pose.data, dtype=np.int16)
        left_arm_joint_pts = [0, 2]
        right_arm_joint_pts = [1, 3]
        self.arm_points = arm_loc_np.reshape((arm_loc_np.shape[0]//2, -1), order='C')[left_arm_joint_pts].tolist()
        print(self.arm_points)
        # hold the current values
        pre_canvas_pts_3d = self.canvas_pts_3D

        count = 0
        for dt in pc2.read_points(pointcloud, field_names={'x','y','z'}, skip_nans=False, uvs=self.canvas_pts):
            # check if it's not NaN
            if dt[0] == dt[0]:
                # print(f'x: {dt[0]}, y: {dt[1]}, z: {dt[2]}')
                self.canvas_pts_3D[count, :] = dt
            else:
                self.canvas_pts_3D = pre_canvas_pts_3d
                break
            count += 1
        
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
        print('end')

    def callback(self, forearm_pose, pointcloud):
        self.update_points(forearm_pose, pointcloud)
        try:
            left_arm_line_3D = Line3D(Point3D(*self.arm_points_3D[0, :]), Point3D(*self.arm_points_3D[1, :]))
            canvas_plane = Plane(Point3D(*self.canvas_pts_3D[0, :]), Point3D(*self.canvas_pts_3D[1, :]), Point3D(*self.canvas_pts_3D[2, :]))
            canvas_plane_point = canvas_plane.intersection(left_arm_line_3D)[0].evalf()
            
            self.msg.header.stamp = rospy.Time.now()
            self.msg.header.frame_id = 'kinect2_rgb_optical_frame'
            self.msg.pose.position.x = canvas_plane_point.x
            self.msg.pose.position.y = canvas_plane_point.y
            self.msg.pose.position.z = canvas_plane_point.z
            # Make sure the quaternion is valid and normalized
            self.msg.pose.orientation.x = 0.0
            self.msg.pose.orientation.y = 0.0
            self.msg.pose.orientation.z = 0.0
            self.msg.pose.orientation.w = 1.0
            # self.frame_id += 1
            # print(canvas_plane_point)
            # print(canvas_plane_point.y)
        except ValueError as error:
            rospy.loginfo(error)

        self.pub.publish(self.msg)
        print('msg published')

    def get_points_on_plane(self, bounding_box, pointcloud):
        n_points = 20
        bounding_box = np.array(bounding_box)
        points = np.vstack((self.points_on_triangle(bounding_box[:3], n_points),self.points_on_triangle(bounding_box[1:], n_points))).astype(int).tolist()

        count = 0
        points_3d = np.zeros((2,3))
        for dt in pc2.read_points(pointcloud, field_names={'x','y','z'}, skip_nans=True, uvs=points):
            # check if it's not NaN
            # if dt[0] == dt[0]:
                # print(f'x: {dt[0]}, y: {dt[1]}, z: {dt[2]}')
            points_3d[count] = dt
            count += 1
            if count is 2:
                break
        
        return [True, points_3d] if count is 2 else [False, None]
            
    def points_on_triangle(self, v, n):
        '''
        Generates uniformly distributed points on a given triangle.

        Parameters
        ----------
        v : ndarray
            [description]
        n : int
            [description]

        Returns
        -------
        [type]
            [description]
        '''
        x = np.sort(np.random.rand(2, n), axis=0)
        return np.column_stack([x[0], x[1]-x[0], 1.0-x[1]]) @ v



def read_depth(width, height, data) :
    # read function
    if (height >= data.height) or (width >= data.width) :
        return -1
    data_out = pc2.read_points(data, field_names=None, skip_nans=False, uvs=[[width, height]])
    int_data = next(data_out)
    rospy.loginfo("int_data " + str(int_data))
    return int_data


if __name__ == '__main__':
    canvas_pts=[[850, 520],[875, 520],[800, 450]]
    CanvasInteraction(canvas_pts=canvas_pts)
    # try:
    #     listen()
    # except rospy.ROSInterruptException as error:
    #     rospy.loginfo(error)