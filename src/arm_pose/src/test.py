#!/usr/bin/env python

# from sympy import Point3D, Line, Plane
# import os
# os.chdir(os.path.dirname(os.path.abspath(__file__)))
# print(os.getcwd())
# from time import time
# from fg._3d import Plane, Line3D, Point3D
import numpy as np 
import rospy
# from arm_pose.src.interaction_canvas import CanvasInteraction
import message_filters
from arm_pose.msg import Floats
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
import sensor_msgs.point_cloud2 as pc2


class ObjectInteraction:
    def __init__(self):
        """
        Initializes the class and starts the subscribers.

        Args:
            canvas_pts (`ndarray`): Three pixel locations (co-planar in 3D space) on an image that describe a surface.
        """
        # self.K is the camera matrix retrieved from /kinect2/qhd/camera_info
        # self.K = np.array([540.68603515625, 0.0, 479.75, 0.0, 540.68603515625, 269.75, 0.0, 0.0, 1.0]).reshape((3,3), order='C')
        # self.canvas_pts = canvas_pts
        # self.canvas_pts_3D = np.zeros((len(self.canvas_pts), 3))

        # initialize arm points in 2D and 3D
        self.arm_points = np.zeros((2, 2), dtype=np.int16).tolist()
        self.arm_points_3D = np.random.random_sample((2, 3))
        self.detected_object_plane = {}
        self.image = None
        # self.t = symbols("t")
        self.compute_once = [True, True]
        # self.compute_once = [False, False]
        self.regress_plane = True

        rospy.init_node("object_selection_node", anonymous=False)

        self.frame_id = "kinect2_rgb_optical_frame"
        # self.point = Point3D(np.zeros((1, 3)))
        self.counter = 0
        # forearm_pose_sub = message_filters.Subscriber('/kinect2/qhd/camera_info', CameraInfo)
        forearm_pose_sub = message_filters.Subscriber("/forearm_pose", Floats)
        pointcloud_sub = message_filters.Subscriber("/kinect2/qhd/points", PointCloud2)
        image_sub = message_filters.Subscriber("/kinect2/qhd/image_color_rect", Image)
        # detected_object_sub = message_filters.Subscriber("/detected_object", String)

        ts = message_filters.ApproximateTimeSynchronizer(
            [forearm_pose_sub, pointcloud_sub, image_sub],
            10,
            1,
            allow_headerless=True,
        )  # Changed code
        ts.registerCallback(self.callback)
        # spin
        rospy.spin()

    def callback(self, forearm_pose, pointcloud, sensor_image):
        image = np.frombuffer(
            sensor_image.data,
            dtype=np.uint8).reshape(
                sensor_image.height,
                sensor_image.width,
                -1)
        if self.image is None:
            self.image = image

        arm_loc_np = np.asarray(forearm_pose.data, dtype=np.int16)
        arm_loc_np = arm_loc_np.reshape((arm_loc_np.shape[0] // 2, -1), order="C")
        # print(arm_loc_np)
        # if arm is "right":
        #     arm_joint_pts = [3, 1]
        # else:
        #     arm_joint_pts = [2, 0]  # 2 is LElbow, 0 is LWrist
        # # right_arm_joint_pts = [0, 2]
        # self.arm_points = arm_loc_np.reshape((arm_loc_np.shape[0] // 2, -1), order="C")[
        #     arm_joint_pts
        # ]
        # arb_arm_points = segment_arb_pts(
        #     self.arm_points,
        #     n_pts=10,
        #     sub_val_range=self.config["canvas_interaction"]["sub_val_range"],
        # )

        arm_points_3d = np.zeros((4, 3))
        for pt_count, dt in enumerate(
            pc2.read_points(
                pointcloud,
                field_names={"x", "y", "z"},
                skip_nans=False,
                uvs=arm_loc_np.tolist(),
            )
        ):
            # print(type(dt))
            # if pt_count == 2:
            #     self.arm_points_3D = pre_arm_points_3D
            #     # count = 0
            #     print("updated")
            #     break 
            # # check if it's not NaN
            # if dt[0] != dt[0]:
            #     print(f'count is: {pt_count}')
            arm_points_3d[pt_count] = dt
        text = ''
        if not np.any(np.isnan(arm_points_3d[[0,2]])):
            # text += 'left\n' if np.abs(arm_points_3d[0, 0]/arm_points_3d[2,0])
            # grad = np.sum(np.abs(np.gradient(arm_points_3d[[0, 2]][:, :2], axis=0)[0]))
            grad = np.gradient(arm_points_3d[[0, 2]], axis=0)
            
            if np.abs(grad[0, 0]) >= 0.15:
                text = 'left hand\n' 
            # print(f'left grad is: {grad}')
            
        if not np.any(np.isnan(arm_points_3d[[1,3]])):
            # text += 'left\n' if np.abs(arm_points_3d[0, 0]/arm_points_3d[2,0])
            # grad = np.sum(np.abs(np.gradient(arm_points_3d[[1,3]][:, :2], axis=0)[0]))
            grad = np.gradient(arm_points_3d[[1, 3]], axis=0)
            if np.abs(grad[0, 0]) >= 0.15:
                text += 'right hand'
            # print(f'right grad is: {grad}')
        # print(arm_points_3d)
        print(text)

if __name__ == '__main__':
    # print("here")

    ObjectInteraction()