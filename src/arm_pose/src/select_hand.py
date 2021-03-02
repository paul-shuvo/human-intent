#!/usr/bin/env python

import numpy as np 
import rospy
# from arm_pose.src.interaction_canvas import CanvasInteraction
import message_filters
from arm_pose.msg import Floats
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
import sensor_msgs.point_cloud2 as pc2
import cv2

# TODO: Add docs to the pointing hand measure
# TODO: Check if it works better with x and y gradiants

class PointingHands:
    def __init__(self):
        """
        Initializes the class and starts the subscribers.

        Args:
            canvas_pts (`ndarray`): Three pixel locations
            (co-planar in 3D space) on an image that describe a surface.
        """

        # initialize arm points in 2D and 3D
        self.arm_points = np.zeros((2, 2), dtype=np.int16).tolist()
        self.arm_points_3D = np.random.random_sample((2, 3))
        self.image = None

        rospy.init_node("pointing_arm_node", anonymous=False)

        self.frame_id = "kinect2_rgb_optical_frame"

        forearm_pose_sub = message_filters.Subscriber("/forearm_pose", Floats)
        pointcloud_sub = message_filters.Subscriber("/kinect2/qhd/points",
                                                    PointCloud2)
        image_sub = message_filters.Subscriber("/kinect2/qhd/image_color_rect",
                                               Image)

        ts = message_filters.ApproximateTimeSynchronizer(
            [forearm_pose_sub, pointcloud_sub, image_sub],
            10,
            1,
            allow_headerless=True,
        )
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
        arm_loc_np = arm_loc_np.reshape((arm_loc_np.shape[0] // 2, -1),
                                        order="C")

        arm_points_3d = np.zeros((4, 3))
        for pt_count, dt in enumerate(
            pc2.read_points(
                pointcloud,
                field_names={"x", "y", "z"},
                skip_nans=False,
                uvs=arm_loc_np.tolist(),
            )
        ):
            arm_points_3d[pt_count] = dt

        text = ''
        if not np.any(np.isnan(arm_points_3d[[0, 2]])):
            grad = np.gradient(arm_points_3d[[0, 2]], axis=0)

            if np.abs(grad[0, 0]) >= 0.15:
                text = 'left hand\n'

        if not np.any(np.isnan(arm_points_3d[[1, 3]])):
            grad = np.gradient(arm_points_3d[[1, 3]], axis=0)
            if np.abs(grad[0, 0]) >= 0.15:
                text += 'right hand'

        if text == 'left hand\nright hand':
            text = 'both hand'

        print(text)
        # cv2.imshow("frame",self.image)
        # cv2.waitKey(30)


if __name__ == '__main__':
    # print("here")

    PointingHands()