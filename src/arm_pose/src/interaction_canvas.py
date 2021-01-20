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

from os.path import dirname, abspath
from os import chdir
import yaml

dir_path = dirname(abspath(__file__))
chdir(dir_path)

class CanvasInteraction:
    def __init__(self, config, topics):  
        """
        Initializes the class and starts the subscribers.

        Args:
            canvas_pts (`ndarray`): Three pixel locations (co-planar in 3D space) on an image that describe a surface.
        """
        self.config = config
        self.canvas_box = self.config['canvas_box']
        self.canvas_pts_3D = np.zeros((3, 3))
        self.arm_points = np.zeros((2, 2), dtype=np.int16).tolist()
        self.arm_points_3D = np.zeros((2, 3))
        self.left_arm_line_3D = None
        self.canvas_plane = None
        self.canvas_plane_point = None
        self.compute_canvas_once = self.config['canvas_interaction']['compute_canvas_once']
        self.arm_pose= PoseStamped()

        
        rospy.init_node('canvas_interaction_node', anonymous=False)
        self.pub = rospy.Publisher('/point_on_canvas', PoseStamped, queue_size=10)
        self.msg = PoseStamped()
        self.frame_id = 'kinect2_rgb_optical_frame'
        # forearm_pose_sub = message_filters.Subscriber('/kinect2/qhd/camera_info', CameraInfo)
        # self.forearm_pose_pub = rospy.Publisher('/forearm_pose_2', PoseStamped, queue_size=10)
        subscribers = [message_filters.Subscriber(topic['topic_name'], topic['data_type']) 
                     for id, topic in topics.items()]
        # forearm_pose_sub = message_filters.Subscriber('/forearm_pose', Floats)
        # pointcloud_sub = message_filters.Subscriber('/kinect2/qhd/points', PointCloud2)      
        ts = message_filters.ApproximateTimeSynchronizer(subscribers, 10, 1, allow_headerless=True) # Changed code
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
        arb_arm_points = segment_arb_pts(self.arm_points, n_pts=10, 
                                         sub_val_range=self.config['canvas_interaction']['sub_val_range'])

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
            self.get_3d_points_on_plane(self.canvas_box, pointcloud)
            self.compute_canvas_once = False
        else:
            self.get_3d_points_on_plane(self.canvas_box, pointcloud)
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

    def get_3d_points_on_plane(self, bounding_box, pointcloud, n_points=10):
        
        bounding_box = np.array(bounding_box)
        points = np.vstack((points_on_triangle(bounding_box[:3], n_points), points_on_triangle(bounding_box[1:], n_points))).astype(int).tolist()
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
   
if __name__ == '__main__':
    with open('config.yml', 'r') as stream:
        config = yaml.safe_load(stream)
    # canvas_pts=np.array([[600, 300], [600, 400], [800, 400], [800, 300]], dtype=np.int32)
    topics = {
        0: {
            'topic_name': '/forearm_pose',
            'data_type': Floats
        },
        1: {
            'topic_name': '/kinect2/qhd/points',
            'data_type': PointCloud2
        }
    }
    CanvasInteraction(config=config, topics=topics)
    
        #     forearm_pose_sub = message_filters.Subscriber('/forearm_pose', Floats)
        # pointcloud_sub = message_filters.Subscriber('/kinect2/qhd/points', PointCloud2)  
    # try:
    #     listen()
    # except rospy.ROSInterruptException as error:
    #     rospy.loginfo(error)