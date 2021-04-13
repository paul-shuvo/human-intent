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

class SelectHand:
    def __init__(self):
        """
        Initializes the class and starts the subscribers.

        Args:
            canvas_pts (`ndarray`): Three pixel locations
            (co-planar in 3D space) on an image that describe a surface.
        """
        self.is_left = 0
        self.is_right = 0
        self.both = 0
        self.inside_left = 0
        self.inside_right = 0
        self.inside_both = 0
        self.both_left = 0
        self.both_right = 0
        ##################################################
        self.paper_data = {
            'frame_total': 0,
            'left_arm': {
                'count': 0,
                'point_left': 0,
                'point_right': 0
            },
            'right_arm': {
                'count': 0,
                'point_left': 0,
                'point_right': 0
            }
        }
        ##################################################
        # initialize arm points in 2D and 3D
        self.arm_points = np.zeros((2, 2), dtype=np.int16).tolist()
        self.arm_points_3D = np.random.random_sample((2, 3))
        self.image = None

        rospy.init_node("pointing_arm_node", anonymous=False)

        # self.frame_id = "kinect2_rgb_optical_frame"
        self.frame_id = "camera_rgb_optical_frame"

        forearm_pose_sub = message_filters.Subscriber("/forearm_pose", Floats)
        # pointcloud_sub = message_filters.Subscriber("/camera/depth_registered/points",
        #                                             PointCloud2)
        # pointcloud_sub = message_filters.Subscriber("/kinect2/qhd/points",
        #                                             PointCloud2)
        # image_sub = message_filters.Subscriber("/camera/rgb/image_color",
        #                                        Image)
        image_sub = message_filters.Subscriber("/kinect2/qhd/image_color_rect",
                                               Image)

        ts = message_filters.ApproximateTimeSynchronizer(
            [forearm_pose_sub, image_sub],
            10,
            1,
            allow_headerless=True,
        )
        ts.registerCallback(self.callback)
        # spin
        rospy.spin()

    def callback(self, forearm_pose, sensor_image):
        self.paper_data['frame_total'] +=1
        image = np.frombuffer(
                    sensor_image.data,
                    dtype=np.uint8).reshape(
                            sensor_image.height,
                            sensor_image.width,
                            -1)
        if self.image is None:
            self.image = image

        arm_loc_np = np.asarray(forearm_pose.data, dtype=np.int16)
        print(arm_loc_np)
        arm_loc_np = arm_loc_np.reshape((arm_loc_np.shape[0] // 2, -1),
                                        order="C")
        print(arm_loc_np)
        is_pointing = False
        if np.abs(arm_loc_np[2] - arm_loc_np[0])[1] < np.abs(arm_loc_np[3] - arm_loc_np[1])[1] * 0.8:
            print(f'Left arm is pointing') 
            is_pointing = True
        elif np.abs(arm_loc_np[2] - arm_loc_np[0])[1] * 0.8 > np.abs(arm_loc_np[3] - arm_loc_np[1])[1]:
            print(f'Right arm is pointing')
            is_pointing = True
        else:
            print(f'No pointing') 

        delta_h_left = np.linalg.norm(arm_loc_np[2]- arm_loc_np[0]) // 2 
        delta_h_right = np.linalg.norm(arm_loc_np[3]- arm_loc_np[1]) // 2 
        # print(delta_h)
        if is_pointing:
            if np.abs(arm_loc_np[2, 0] - arm_loc_np[0, 0]) >= delta_h_left:
                self.paper_data['left_arm']['count'] += 1
                # print(f'Left arm is pointing')
                if arm_loc_np[2, 0] - arm_loc_np[0, 0] < delta_h_left * 0.75:
                    self.paper_data['left_arm']['point_left'] += 1
                    # print(f'Pointing left')
                elif arm_loc_np[2, 0] - arm_loc_np[0, 0] > delta_h_left * 0.75:
                    self.paper_data['left_arm']['point_right'] += 1
                    # print(f'Pointing right')

            elif np.abs(arm_loc_np[3, 0] - arm_loc_np[1, 0]) >= delta_h_right:
                self.paper_data['right_arm']['count'] += 1
                # print(f'Right arm is pointing')
                if arm_loc_np[3, 0] - arm_loc_np[1, 0] < delta_h_right * 0.75:
                    self.paper_data['right_arm']['point_left'] += 1
                    # print(f'Pointing left')
                elif arm_loc_np[3, 0] - arm_loc_np[1, 0] > delta_h_right * 0.75:
                    self.paper_data['right_arm']['point_right'] += 1
                    # print(f'Pointing right')
            else:
                print('Pointing Straight')

        if self.paper_data['frame_total'] % 10 is 0:
            print(self.paper_data)
        
        # arm_points_3d = np.zeros((4, 3))
        # for pt_count, dt in enumerate(
        #     pc2.read_points(
        #         pointcloud,
        #         field_names={"x", "y", "z"},
        #         skip_nans=False,
        #         uvs=arm_loc_np.tolist(),
        #     )
        # ):
        #     arm_points_3d[pt_count] = dt

        # gradient_threshold = 0.05
        # text = ''
        # grad1 = 0
        # grad2 = 0
        # print(f'=============================\nArm points 3D are: \n {arm_points_3d}\n=============================')
        # if not np.any(np.isnan(arm_points_3d[[0, 2]])):
        #     grad = np.gradient(arm_points_3d[[0, 2]], axis=0)
        #     grad1 = grad[0, 0]
        #     self.inside_left += 1
        #     if np.abs(grad[0, 0]) >= gradient_threshold:
        #         text = 'left hand\n'
        #         self.is_left += 1

        # if not np.any(np.isnan(arm_points_3d[[1, 3]])):
        #     grad = np.gradient(arm_points_3d[[1, 3]], axis=0)
        #     grad2 = grad[0, 0]
        #     self.inside_right += 1
        #     print(np.abs(grad[0, 0]))
        #     if np.abs(grad[0, 0]) >= gradient_threshold:
        #         text += 'right hand'
        #         self.is_right += 1

        # if text == 'left hand\nright hand':
        #     # print('==========================================')
        #     if grad1 >= grad2:
        #         self.both_left += 1
        #         # print('left')
        #     else:
        #         self.both_right += 1
        #         # print('right')
        #     # print('==========================================')
        #     self.both += 1
        #     text = 'both hand'

        # print(f'inside left: {self.inside_left}, is left: {self.is_left}')
        # print(f'inside right: {self.inside_right}, is right: {self.is_right}')
        # print(f'is both: {self.both}')
        # print(f'is both left: {self.both_left}, is both right: {self.both_right}')

        # cv2.imshow("frame",self.image)
        # cv2.waitKey(30)


if __name__ == '__main__':
    # print("here")

    SelectHand()