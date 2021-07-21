#!/usr/bin/env python

import numpy as np 
import rospy
import ros_numpy

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
        self.image_pub = rospy.Publisher('/operating_hand',
                                              Image, queue_size=10)

        forearm_pose_sub = message_filters.Subscriber("/forearm_pose", Floats)
        # pointcloud_sub = message_filters.Subscriber("/camera/depth_registered/points",
        #                                             PointCloud2)
        # pointcloud_sub = message_filters.Subscriber("/kinect2/qhd/points",
        #                                             PointCloud2)
        image_sub = message_filters.Subscriber("/camera/rgb/image_color",
                                               Image)
        # image_sub = message_filters.Subscriber("/kinect2/qhd/image_color_rect",
                                            #    Image)

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
        # print(arm_loc_np)
        arm_loc_np = arm_loc_np.reshape((arm_loc_np.shape[0] // 2, -1),
                                        order="C")
        # print(arm_loc_np)
        is_pointing = False
        operating_hand = None

        if np.abs(arm_loc_np[2] - arm_loc_np[0])[1] < np.abs(arm_loc_np[3] - arm_loc_np[1])[1] * 0.8:
            print(f'Left arm is pointing') 
            is_pointing = True
            operating_hand = 'Left arm'
        elif np.abs(arm_loc_np[2] - arm_loc_np[0])[1] * 0.8 > np.abs(arm_loc_np[3] - arm_loc_np[1])[1]:
            print(f'Right arm is pointing')
            is_pointing = True
            operating_hand = 'Right arm'
        else:
            print(f'No pointing') 

        delta_h_left = np.linalg.norm(arm_loc_np[2]- arm_loc_np[0]) // 2 
        delta_h_right = np.linalg.norm(arm_loc_np[3]- arm_loc_np[1]) // 2 
        pointing_direction = None
        # print(delta_h)
        if is_pointing:
            if np.abs(arm_loc_np[2, 0] - arm_loc_np[0, 0]) >= delta_h_left:
                self.paper_data['left_arm']['count'] += 1
                # print(f'Left arm is pointing')
                if arm_loc_np[2, 0] - arm_loc_np[0, 0] < delta_h_left * 0.75:
                    self.paper_data['left_arm']['point_left'] += 1
                    # print(f'Pointing left')
                    pointing_direction = 'Left'
                elif arm_loc_np[2, 0] - arm_loc_np[0, 0] > delta_h_left * 0.75:
                    self.paper_data['left_arm']['point_right'] += 1
                    # print(f'Pointing right')
                    pointing_direction = 'right'

            elif np.abs(arm_loc_np[3, 0] - arm_loc_np[1, 0]) >= delta_h_right:
                self.paper_data['right_arm']['count'] += 1
                # print(f'Right arm is pointing')
                if arm_loc_np[3, 0] - arm_loc_np[1, 0] < delta_h_right * 0.75:
                    self.paper_data['right_arm']['point_left'] += 1
                    # print(f'Pointing left')
                    pointing_direction = 'Left'
                elif arm_loc_np[3, 0] - arm_loc_np[1, 0] > delta_h_right * 0.75:
                    self.paper_data['right_arm']['point_right'] += 1
                    # print(f'Pointing right')
                    pointing_direction = 'Right'
            else:
                print('Pointing Straight')

        if self.paper_data['frame_total'] % 10 is 0:
            print(self.paper_data)
        image_str = f'Is pointing: {"Yes" if is_pointing else "No"}\nPointing hand: {"None" if operating_hand is None else operating_hand}\nPointing direction: {"None" if pointing_direction is None else pointing_direction}'
        image_str = image_str.split('\n')
        # for i in range(len(image_str)):
        #     image = cv2.putText(image, image_str[i], (600, 200 + int(40*i)), 
        #                         cv2.FONT_HERSHEY_SIMPLEX, 0.75, 
        #                         (255, 0, 0), 1, cv2.LINE_AA)
        
        image = cv2.rectangle(image, (530, 300), (765, 400), (55,55,180), 2)
        points = np.array([[530, 300], [530, 400], [765, 400], [765, 300]],
                  dtype=np.int32)

        cv2.fillPoly(image, [points], (100, 180, 150))
        for i in range(len(image_str)):
            image = cv2.putText(image, image_str[i], (550, 325 + int(25*i)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                                (255, 0, 0), 1, cv2.LINE_AA)
        image = ros_numpy.msgify(Image, image, encoding='bgr8')
        self.image_pub.publish(image)

if __name__ == '__main__':
    # print("here")

    SelectHand()