#!/usr/bin/env python
from sklearn.utils.validation import _num_samples
import rospy
from arm_pose.msg import Floats

# import ros_np_multiarray as rnm
import numpy as np
np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)
import json
import cv2

# from rospy.numpy_msg import numpy_msg
import message_filters
from std_msgs.msg import String
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from geometry_msgs.msg import PoseStamped, Pose, PoseArray

# from sympy import Point3D, Plane, symbols, N
# from sympy.geometry import Line3D, Segment, Ray3D

from time import time
from fg._3D import Point3D, Line3D, Plane

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
        self.object_center = {}
        # self.t = symbols("t")
        # self.compute_once = [True, True]
        self.compute_once = [False, False]
        self.regress_plane = False
        self.arr = np.zeros((3, 4, 2))
        ####### Experiment Variable #######
        self.detection_count = {
            'cheeze-it': 0,
            'book-2': 0,
            'book-3': 0
        }
        self.detection_voting_count = {
            'cheeze-it': 0,
            'book-2': 0,
            'book-3': 0
        }
        self.voting_interval = 0
        self.detection_voting = {
            'cheeze-it': 0,
            'book-2': 0,
            'book-3': 0
        }
        ####### Experiment Variable #######

        rospy.init_node("object_selection_node", anonymous=False)
        # self.pub = rospy.Publisher("/object_selected", PoseStamped, queue_size=10)
        # self.msg = PoseStamped()

        self.pose_array_pub = rospy.Publisher('/object_pose_array',
                                              PoseArray, queue_size=10)
        self.pose_array = PoseArray()

        self.frame_id = "camera_rgb_optical_frame"
        self.point = Point3D(np.zeros((1, 3)))

        self.counter = 0
        # forearm_pose_sub = message_filters.Subscriber('/kinect2/qhd/camera_info', CameraInfo)
        forearm_pose_sub = message_filters.Subscriber("/forearm_pose", Floats)
        # pointcloud_sub = message_filters.Subscriber("/camera/depth_registered/points", PointCloud2)
        detected_object_sub = message_filters.Subscriber("/detected_object", String)
        self.image_sub = message_filters.Subscriber(
                            "/camera/rgb/image_rect_color", Image
                        )
        ts = message_filters.ApproximateTimeSynchronizer(
            [forearm_pose_sub, detected_object_sub, self.image_sub],
            10,
            1,
            allow_headerless=True,
        )  # Changed code
        ts.registerCallback(self.callback)
        # spin
        rospy.spin()

    def callback(self, forearm_pose, detected_object, image):
        frame = np.frombuffer(image.data, dtype=np.uint8).reshape(
            image.height, image.width, -1
        )
        arm_loc_np = np.asarray(forearm_pose.data, dtype=np.int16)
        self.arm_points = arm_loc_np.reshape((arm_loc_np.shape[0] // 2, -1), order="C")

        try:
            self.arr = np.array(list(json.loads(detected_object.data).values()))
        except:
            print('Error in parsing object boundary')
        # print(arr.shape)
        # print(arr)
        arr = self.arr
        min_x = np.min(arr[:, :, 0]).astype(np.int16) if len(arr.shape) is 3 else np.min(arr[ :, 0]).astype(np.int16)
        max_x = np.max(arr[:, :, 0]).astype(np.int16) if len(arr.shape) is 3 else np.min(arr[ :, 0]).astype(np.int16)
        y = np.mean(arr[:, :, 1]).astype(np.int16) if len(arr.shape) is 3 else np.mean(arr[ :, 1]).astype(np.int16)
        frame = cv2.line(
            frame,
            tuple([min_x, y]),
            tuple([max_x, y]),
            (255, 0, 255),
            2,
        )
        # print(frame.dtype)
        # cv2.imshow('Visualize', frame)
        # cv2.waitKey()
        
        cv2.startWindowThread()
        cv2.namedWindow("preview")
        cv2.imshow("preview", frame)
        cv2.waitKey(10)



def read_depth(width, height, data):
    # read function
    if (height >= data.height) or (width >= data.width):
        return -1
    data_out = pc2.read_points(
        data, field_names=None, skip_nans=False, uvs=[[width, height]]
    )
    int_data = next(data_out)
    rospy.loginfo("int_data " + str(int_data))
    return int_data


if __name__ == "__main__":
    ObjectInteraction()
    # try:
    #     listen()
    # except rospy.ROSInterruptException as error:
    #     rospy.loginfo(error)
