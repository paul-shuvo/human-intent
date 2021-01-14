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
from utils import segment_arb_pts, points_on_triangle


class CanvasInteraction:
    def __init__(self, canvas_box):  
        """
        Initializes the class and starts the subscribers.

        Args:
            canvas_pts (`ndarray`): Three pixel locations (co-planar in 3D space) on an image that describe a surface.
        """
        # self.K is the camera matrix retrieved from /kinect2/qhd/camera_info 
        self.K = np.array([540.68603515625, 0.0, 479.75, 0.0, 540.68603515625, 269.75, 0.0, 0.0, 1.0]).reshape((3,3), order='C')
        self.canvas_box = canvas_box
        self.canvas_pts_3D = np.zeros((3, 3))
        self.arm_points = np.zeros((2, 2), dtype=np.int16).tolist()
        self.arm_points_3D = np.zeros((2, 3))
        self.left_arm_line_3D = None
        self.canvas_plane = None
        self.canvas_plane_point = None
        self.compute_canvas_once = False
        self.arm_pose= PoseStamped()

        
        rospy.init_node('projection_node', anonymous=False)
        self.pub = rospy.Publisher('/point_on_canvas', PoseStamped, queue_size=10)
        self.msg = PoseStamped()
        self.frame_id = 'kinect2_rgb_optical_frame'
        # forearm_pose_sub = message_filters.Subscriber('/kinect2/qhd/camera_info', CameraInfo)
        # self.forearm_pose_pub = rospy.Publisher('/forearm_pose_2', PoseStamped, queue_size=10)

        forearm_pose_sub = message_filters.Subscriber('/forearm_pose', Floats)
        pointcloud_sub = message_filters.Subscriber('/kinect2/qhd/points', PointCloud2)      
        ts = message_filters.ApproximateTimeSynchronizer([forearm_pose_sub, pointcloud_sub], 10, 1, allow_headerless=True) # Changed code
        ts.registerCallback(self.callback)
        # spin
        rospy.spin()
    
    def pose_from_vector3D(self, position, waypoint):
        #http://lolengine.net/blog/2013/09/18/beautiful-maths-quaternion-from-vectors
        pass
            
    def update_points_arm_3d(self, forearm_pose, pointcloud, arm='left'):
        arm_loc_np = np.asarray(forearm_pose.data, dtype=np.int16)
        if arm is 'right':
            arm_joint_pts = [3, 1]
        else:
            arm_joint_pts = [2, 0] # 2 is LElbow, 0 is LWrist
        # right_arm_joint_pts = [0, 2]
        self.arm_points = arm_loc_np.reshape((arm_loc_np.shape[0]//2, -1), order='C')[arm_joint_pts]
        arb_arm_points = segment_arb_pts(self.arm_points, n_pts=10)

        pre_arm_points_3D = self.arm_points_3D
        for pt_count, dt in enumerate(pc2.read_points(pointcloud, field_names={'x','y','z'}, skip_nans=True, uvs=arb_arm_points.astype(int).tolist()[5:10])):
            # print(type(dt))
            if pt_count == 2:
                self.arm_points_3D = pre_arm_points_3D
                # count = 0
                print('updated')
                break
            # check if it's not NaN
            # if dt[0] == dt[0]:
                # print(f'x: {dt[0]}, y: {dt[1]}, z: {dt[2]}')
            pre_arm_points_3D[pt_count, :] = dt
            print(f'pre: {pre_arm_points_3D}')
            print(f'post:{self.arm_points_3D}')
            # count += 1

        # print(self.arm_points_3D)
        # print('end')


    def callback(self, forearm_pose, pointcloud):
        
        if self.compute_canvas_once:
            self.points_3d_on_plane(self.canvas_box, pointcloud)
            self.compute_canvas_once = False
        else:
            self.points_3d_on_plane(self.canvas_box, pointcloud)
        print(f'Plane is {self.canvas_plane}')
        self.update_points_arm_3d(forearm_pose, pointcloud)
        # self.pose_from_vector3D(self.arm_points_3D[0], self.arm_points_3D[1] - self.arm_points_3D[0])
        
        try:
            self.left_arm_line_3D = Line3D(Point3D(*self.arm_points_3D[0, :]), Point3D(*self.arm_points_3D[1, :]))
            self.canvas_plane = Plane(Point3D(*self.canvas_pts_3D[0, :]), Point3D(*self.canvas_pts_3D[1, :]), Point3D(*self.canvas_pts_3D[2, :]))
        except ValueError as error:
            rospy.loginfo(error)
            
        self.canvas_plane_point = self.canvas_plane.intersection(self.left_arm_line_3D)[0].evalf()
        
        self.msg.header.stamp = rospy.Time.now()
        self.msg.header.frame_id = 'kinect2_rgb_optical_frame'
        self.msg.pose.position.x = self.canvas_plane_point.x
        self.msg.pose.position.y = self.canvas_plane_point.y
        self.msg.pose.position.z = self.canvas_plane_point.z
        # Make sure the quaternion is valid and normalized
        self.msg.pose.orientation.x = 0.0
        self.msg.pose.orientation.y = 0.0
        self.msg.pose.orientation.z = 0.0
        self.msg.pose.orientation.w = 1.0
            # self.frame_id += 1
            # print(canvas_plane_point)
            # print(canvas_plane_point.y)


        self.pub.publish(self.msg)
        print('msg published')

    def points_3d_on_plane(self, bounding_box, pointcloud, n_points=10):
        
        bounding_box = np.array(bounding_box)
        points = np.vstack((self.points_on_triangle(bounding_box[:3], n_points),self.points_on_triangle(bounding_box[1:], n_points))).astype(int).tolist()
        # points = self.canvas_box
        print(f'Points are:\n{points}')
        # count = 0
        points_3d = np.zeros((3,3))
        for pt_count, dt in enumerate(pc2.read_points(pointcloud, field_names={'x','y','z'}, skip_nans=True, uvs=points)):
            # check if it's not NaN
            # if dt[0] == dt[0]:
                # print(f'x: {dt[0]}, y: {dt[1]}, z: {dt[2]}')
            if pt_count is 3:
                self.canvas_pts_3D = points_3d
                break
            points_3d[pt_count] = dt
            
        self.canvas_plane = Plane(Point3D(*self.canvas_pts_3D[0, :]), Point3D(*self.canvas_pts_3D[1, :]), Point3D(*self.canvas_pts_3D[2, :]))

        # return [True, points_3d] if count is 2 else [False, None]
            
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
    canvas_pts=np.array([[600, 300], [600, 400], [800, 400], [800, 300]], dtype=np.int32)
    CanvasInteraction(canvas_box=canvas_pts)
    # try:
    #     listen()
    # except rospy.ROSInterruptException as error:
    #     rospy.loginfo(error)