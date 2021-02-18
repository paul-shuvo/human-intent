#!/usr/bin/env python
import rospy
from arm_pose.msg import Floats

# import ros_np_multiarray as rnm
import numpy as np
from rospy.numpy_msg import numpy_msg
from image_geometry import PinholeCameraModel
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped

import cv2
import os

from utils import project3dToPixel

l_pair = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),  # Head
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (17, 11),
    (17, 12),  # Body
    (11, 13),
    (12, 14),
    (13, 15),
    (14, 16),
]

l_pair_by_name = {"l_hand": l_pair[6], "r_hand": l_pair[8]}

p_color = [
    (0, 255, 255),
    (0, 191, 255),
    (0, 255, 102),
    (0, 77, 255),
    (0, 255, 0),  # Nose, LEye, REye, LEar, REar
    (77, 255, 255),
    (77, 255, 204),
    (77, 204, 255),
    (191, 255, 77),
    (77, 191, 255),
    (191, 255, 77),  # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
    (204, 77, 255),
    (77, 255, 204),
    (191, 77, 255),
    (77, 255, 191),
    (127, 77, 255),
    (77, 255, 127),
    (0, 255, 255),
]  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck

line_color = [
    (0, 215, 255),
    (0, 255, 204),
    (0, 134, 255),
    (0, 255, 50),
    (77, 255, 222),
    (77, 196, 255),
    (77, 135, 255),
    (191, 255, 77),
    (77, 255, 77),
    (77, 222, 255),
    (255, 156, 127),
    (0, 127, 255),
    (255, 127, 77),
    (0, 77, 255),
    (255, 77, 36),
]

# TODO: Pass the topic names in the __init__ method


class ShowForearmPose:
    def __init__(self):
        # self.count = 0
        # init the node
        rospy.init_node("viusalize_forearm_node", anonymous=False)
        camera_info_topic = "/kinect2/qhd/camera_info"
        # camera_info = rospy.wait_for_message(camera_info_topic, CameraInfo)
        # print(camera_info)
        self.image = np.zeros((546, 420, 3), dtype=np.uint8)
        self.point = (0, 0)
        # self.data = None
        self.arm_points = np.zeros((4, 2), dtype=np.int16)
        self.counter = 0
        self.cor_x = np.zeros(18)
        self.cor_y = np.zeros(18)
        self.pred_score = np.zeros(18)
        self.part_line = {}

        self.image_sub = message_filters.Subscriber(
            "/kinect2/qhd/image_color_rect", Image
        )
        self.forearm_pose_sub = message_filters.Subscriber("/forearm_pose", Floats)
        self.selected_point_sub = message_filters.Subscriber(
            "/object_selected", PoseStamped
        )
        # if show_directed_point:
        # self.camera_model = PinholeCameraModel()
        # self.camera_model.fromCameraInfo(camera_info)

        ts = message_filters.ApproximateTimeSynchronizer(
            [self.forearm_pose_sub, self.image_sub, self.selected_point_sub],
            10,
            1,
            allow_headerless=True,
        )  # Changed code
        # else:
        #     ts = message_filters.ApproximateTimeSynchronizer([self.forearm_pose_sub, self.image_sub], 10, 1, allow_headerless=True)

        ts.registerCallback(self.callback)
        # spin
        rospy.spin()

    def callback(self, arm_loc, image, selected_point_sub):
        frame = np.frombuffer(image.data, dtype=np.uint8).reshape(
            image.height, image.width, -1
        )
        arm_loc_np = np.asarray(arm_loc.data, dtype=np.int16)
        self.arm_points = arm_loc_np.reshape((arm_loc_np.shape[0] // 2, -1), order="C")

        frame = cv2.line(
            frame,
            tuple(self.arm_points[0]),
            tuple(self.arm_points[2]),
            line_color[2],
            2,
        )
        frame = cv2.line(
            frame,
            tuple(self.arm_points[1]),
            tuple(self.arm_points[3]),
            line_color[2],
            2,
        )
        # frame = cv2.circle(
        #     frame,
        #     (int(self.point[0]), int(self.point[1])),
        #     radius=15,
        #     color=(0, 0, 255),
        #     thickness=5,
        # )
        # frame = cv2.polylines(
        #     frame, [canvas_pts], isClosed=True, color=(255, 0, 0), thickness=2
        # )
        # cv2.putText(
        #     frame,
        #     "canvas plane",
        #     (600, 280),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.5,
        #     (155, 100, 80),
        #     1,
        # )

        cv2.imshow("image", frame)
        cv2.waitKey(30)


class ShowImage(object):
    """Get 3D values of bounding boxes returned by face_recognizer node.

    _bridge (CvBridge): Bridge between ROS and CV image
    pub (Publisher): Publisher object for face depth results
    f (Float): Focal Length
    cx (Int): Principle Point Horizontal
    cy (Int): Principle Point Vertical

    """

    def __init__(self):
        super(ShowImage, self).__init__()
        # self.K = np.array([540.68603515625, 0.0, 479.75, 0.0, 540.68603515625, 269.75, 0.0, 0.0, 1.0]).reshape((3,3), order='C')

        self.count = 0
        # init the node
        rospy.init_node("show_image_node", anonymous=False)
        camera_info_topic = "/kinect2/qhd/camera_info"
        camera_info = rospy.wait_for_message(camera_info_topic, CameraInfo)
        # print(camera_info)
        self.image = np.zeros((546, 420, 3), dtype=np.uint8)
        self.point = (0, 0)
        # self.data = None
        self.arm_points = np.zeros((4, 2), dtype=np.int16)
        self.counter = 0
        self.cor_x = np.zeros(18)
        self.cor_y = np.zeros(18)
        self.pred_score = np.zeros(18)
        self.part_line = {}
        self.frame = None

        self.image_sub = message_filters.Subscriber(
            "/kinect2/qhd/image_color_rect", Image
        )
        self.forearm_pose_sub = message_filters.Subscriber("/forearm_pose", Floats)

        # if show_directed_point:
        self.camera_model = PinholeCameraModel()
        self.camera_model.fromCameraInfo(camera_info)

        # self.selected_point_sub = message_filters.Subscriber('/object_selected', PoseStamped)
        self.selected_point_sub = message_filters.Subscriber(
            "/object_selected", PoseStamped
        )
        
        # if self.viz_pose:
            # self.viz_frame = None
            # camera_info = rospy.wait_for_message(
            #         "/kinect2/qhd/camera_info",
            #         CameraInfo
            #     )
        self.P = np.array(camera_info.P).reshape((3, 4))

        # self.camera_info_sub = message_filters.Subscriber('/kinect2/qhd/camera_info', CameraInfo)

        ts = message_filters.ApproximateTimeSynchronizer(
            [self.forearm_pose_sub, self.image_sub, self.selected_point_sub],
            10,
            1,
            allow_headerless=True,
        )  # Changed code
        # else:
        #     ts = message_filters.ApproximateTimeSynchronizer([self.forearm_pose_sub, self.image_sub], 10, 1, allow_headerless=True)

        ts.registerCallback(self.callback)
        # spin
        rospy.spin()

    def callback(self, arm_loc, image, selected_point_sub):
        print('here')
        # canvas_pts = np.array(
        #     [[600, 300], [600, 400], [800, 400], [800, 300]], dtype=np.int32
        # )
        # if selected_point_sub is not None:
        # self.camera_model.fromCameraInfo(camera_info)
        position = selected_point_sub.pose.position
        point_3d = [position.x, position.y, position.z]
        # self.point = self.camera_model.project3dToPixel(point_3d)
        self.point = project3dToPixel(self.P, point_3d)
        print(self.point)

        frame = np.frombuffer(image.data, dtype=np.uint8).reshape(
            image.height, image.width, -1
        )
        
        if self.frame is None:
            self.frame = frame
        arm_loc_np = np.asarray(arm_loc.data, dtype=np.int16)
        self.arm_points = arm_loc_np.reshape((arm_loc_np.shape[0] // 2, -1), order="C")
        frame = cv2.line(
            frame,
            tuple(self.arm_points[0]),
            tuple(self.arm_points[2]),
            line_color[2],
            2,
        )
        frame = cv2.line(
            frame,
            tuple(self.arm_points[1]),
            tuple(self.arm_points[3]),
            line_color[2],
            2,
        )
        frame = cv2.circle(
            frame,
            (int(self.point[0]), int(self.point[1])),
            radius=15,
            color=(0, 0, 255),
            thickness=5,
        )
        # frame = cv2.polylines(
        #     frame, [canvas_pts], isClosed=True, color=(255, 0, 0), thickness=2
        # )
        # cv2.putText(
        #     frame,
        #     "canvas plane",
        #     (600, 280),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.5,
        #     (155, 100, 80),
        #     1,
        # )
        self.frame = frame
        print('here 2')
        cv2.imshow("image", self.frame)
        cv2.waitKey(30)


# def main():
#     """ main function
#     """
#     node = ShowImage()

if __name__ == "__main__":

    # print(camera_info)
    ShowImage()
    # ShowForearmPose()

# -------------------------
# old code
# -------------------------

# def callback(self, loc, image):
#     frame = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, -1)

#     self.image = frame
#     np_data = np.array(loc.data)

#     if self.cor_x.all() == np_data[0:18].all():
#         print('same')
#     else:
#         self.cor_x, self.cor_y, self.pred_score = np_data[0:18], np_data[18:36], np_data[36:54]

#     # part_line = {}
#     height, width = frame.shape[:2]
#     # img = cv2.resize(frame,(int(width/2), int(height/2)))

#     for n in range(self.pred_score.shape[0]):
#         # print(f'pred score for {n} is: {self.pred_score[n]}')
#         if self.pred_score[n] <= 0.05:
#             continue
#         # cor_x, cor_y = int(kp_preds[n, 0]), int(kp_preds[n, 1])
#         # part_line[n] = (int(cor_x[n]/2), int(cor_y[n]/2))
#         self.part_line[n] = (int(self.cor_x[n]), int(self.cor_y[n]))
#         # print(part_line[n])
#         # bg = img.copy()
#         # cv2.circle(bg, (int(cor_x[n]/2), int(cor_y[n]/2)), 5, p_color[n], -1)
#         cv2.circle(self.image, (int(self.cor_x[n]), int(self.cor_y[n])), 5, p_color[n], -1)

#         # Now create a mask of logo and create its inverse mask also
#         transparency = float(max(0, min(1, self.pred_score[n])))
#         # self.image = cv2.addWeighted(bg, transparency, img, 1-transparency, 0)
#         # self.image = bg
#     try:
#         self.image = cv2.line(self.image, self.part_line[l_pair[6][0]], self.part_line[l_pair[6][1]], line_color[2], 2)
#         self.image = cv2.line(self.image, self.part_line[l_pair[8][0]], self.part_line[l_pair[8][1]], line_color[4], 2)
#         print(self.part_line[6])
#         print(f'length is: {len(self.part_line)}')
#     except KeyError:
#         print("Key Error")
#     print(self.count)
#     cv2.imshow('image', self.image)
#     cv2.waitKey(30)
