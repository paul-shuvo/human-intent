#!/usr/bin/env python
from sklearn.utils.validation import _num_samples
import rospy
from arm_pose.msg import Floats

# import ros_np_multiarray as rnm
import numpy as np
np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)
import json
import cv2
import ros_numpy

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
from fg._2D import Line

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
        self.total_frame = 0
        self.detection_count_alt = {
            'cheez-it': 0,
            'book-2': 0,
            'book-3': 0
        }
        self.detection_count = {
            'cheez-it': 0,
            'book-2': 0,
            'book-3': 0
        }
        self.detection_voting_count = {
            'cheez-it': 0,
            'book-2': 0,
            'book-3': 0
        }
        self.voting_interval = 0
        self.detection_voting = {
            'cheez-it': 0,
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

        self.image_pub = rospy.Publisher('/selected_object',
                                              Image, queue_size=10)
        # self.pose_array = PoseArray()
        
        self.frame_id = "camera_rgb_optical_frame"
        self.point = Point3D(np.zeros((1, 3)))

        self.counter = 0
        # forearm_pose_sub = message_filters.Subscriber('/kinect2/qhd/camera_info', CameraInfo)
        forearm_pose_sub = message_filters.Subscriber("/forearm_pose", Floats)
        # pointcloud_sub = message_filters.Subscriber("/camera/depth_registered/points", PointCloud2)
        detected_object_sub = message_filters.Subscriber("/detected_object", String)
        self.image_sub = message_filters.Subscriber(
                            "/kinect2/qhd/image_color", Image
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
        object_dict = json.loads(detected_object.data)
        try:
            self.arr = np.array(list(object_dict.values()))
        except:
            print('Error in parsing object boundary')
        # print(arr.shape)
        # print(arr)
        arr = self.arr
        dist_ = np.zeros((arr.shape[0], arr.shape[1]))
        # object_dict = json.loads(detected_object.data)
        # print(arr.shape)
        object_center = (arr[:, 0, :] + arr[:, 2, :]) / 2
        # min_x = np.min(arr[:, :, 0]).astype(np.int16) if len(arr.shape) is 3 else np.min(arr[ :, 0]).astype(np.int16)
        # max_x = np.max(arr[:, :, 0]).astype(np.int16) if len(arr.shape) is 3 else np.min(arr[ :, 0]).astype(np.int16)
        min_x = 0
        max_x = frame.shape[1]
        y = np.mean(arr[:, :, 1]).astype(np.int16) if len(arr.shape) is 3 else np.mean(arr[ :, 1]).astype(np.int32)
        frame = cv2.line(
            frame,
            tuple([min_x, y]),
            tuple([max_x, y]),
            (255, 0, 255),
            2,
        )
        
        frame = cv2.line(
            frame,
            tuple(self.arm_points[0]),
            tuple(self.arm_points[2]),
            (100, 187, 255),
            2,
        )
        
        frame = cv2.line(
            frame,
            tuple(self.arm_points[1]),
            tuple(self.arm_points[3]),
            (100, 187, 255),
            2,
        )
        # print(frame.dtype)
        # cv2.imshow('Visualize', frame)
        # cv2.waitKey()
        # arm_line = Line(self.arm_points[[0,2]])
        arm_line = Line(self.arm_points[[1,3]])
        
        for i in range(dist_.shape[0]):
            for j in range(dist_.shape[1]):
                bbox_line = Line(arr[i,[j, j+1]]) if j < 3 else Line(arr[i, [j, 0]])
                bbox_intersection = arm_line.intersect(bbox_line).astype(np.int32)
                # print(bbox_intersection)
                # print(object_center)
                dist_[i,j] = np.linalg.norm(object_center[i] - bbox_intersection)
        
        # print(dist_)
        so = list(object_dict.keys())[np.argmin(np.min(dist_, axis=1))]
        self.detection_count_alt[so] += 1
        print(f'algo 2: {self.detection_count_alt}')

        
        
        # eye_line = Line(self.arm_points[[5,3]])
        object_line = Line(np.array([[min_x, y], [max_x, y]]))
        intersection_arm_line = object_line.intersect(arm_line).astype(np.int32)
        # intersection_eye_line = object_line.intersect(eye_line).astype(np.uint16)
        object_distance = np.linalg.norm(object_center - intersection_arm_line, axis=1)
        selected_object = list(object_dict.keys())[np.argmin(object_distance)] 
        self.total_frame += 1
        self.detection_count[selected_object] += 1
        # print(f'intersection arm line: {intersection_arm_line}')
        print(f'algo 1: {self.detection_count}')
        frame = cv2.line(
            frame,
            tuple(self.arm_points[3]),
            tuple(intersection_arm_line),
            (155, 100, 200),
            2,
        )
        
        # frame = cv2.line(
        #     frame,
        #     tuple(self.arm_points[3]),
        #     tuple(intersection_eye_line),
        #     (50, 120, 255),
        #     2,
        # )
        
        frame = cv2.circle(frame, tuple(intersection_arm_line), radius=10, color=(255, 0, 0), thickness=-1)
        # frame = cv2.circle(frame, tuple(intersection_eye_line), radius=10, color=(0, 255, 0), thickness=-1)

        msg = ros_numpy.msgify(Image, frame, encoding='bgr8')
        self.image_pub.publish(msg)
        # cv2.startWindowThread()
        # cv2.namedWindow("preview")
        # cv2.imshow("preview", frame)
        # cv2.waitKey(10)



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
