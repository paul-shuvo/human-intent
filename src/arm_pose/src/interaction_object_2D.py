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
from utils import draw_angled_text

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

import spacy
from spacy.matcher import Matcher
nlp = spacy.load('en_core_web_sm')
nlp.Defaults.stop_words -= {'give', 'Give', 'put', 'Put', 'it',
                            'this', 'This', 'that', 'That',
                            'here', 'Here', 'there', 'There'}
# nlp.Defaults.stop_words += {'thing'}

matcher = Matcher(nlp.vocab)

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
            'book-1': 0,
            'book-3': 0
        }
        self.detection_count = {
            'cheez-it': 0,
            'book-2': 0,
            'book-1': 0,
            'book-3': 0
        }
        self.detection_voting_count = {
            'cheez-it': 0,
            'book-2': 0,
            'book-1': 0,
            'book-3': 0
        }
        self.voting_interval = 0
        self.detection_voting = {
            'cheez-it': 0,
            'book-2': 0,
            'book-1': 0,
            'book-3': 0
        }
        
        nlp = spacy.load('en_core_web_sm')
        nlp.Defaults.stop_words -= {'give', 'Give', 'put', 'Put', 'it',
                                    'this', 'This', 'that', 'That',
                                    'here', 'Here', 'there', 'There'}
        # nlp.Defaults.stop_words += {'thing'}

        self.matcher = Matcher(nlp.vocab)
        self.object_attr = {
            'cheez-it': {
                'attribute': ['red'],
                'position': 'left'
            },
            'book-1': {
                'attribute': ['blue'],
                'position': 'center'
            },
            'book-2': {
                'attribute': ['red'],
                'position': 'right'
            }
        }
        ####### Experiment Variable #######

        rospy.init_node("object_selection_node", anonymous=False)
        # self.pub = rospy.Publisher("/object_selected", PoseStamped, queue_size=10)
        # self.msg = PoseStamped()

        # self.pose_array_pub = rospy.Publisher('/object_pose_array',
        #                                       PoseArray, queue_size=10)
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
        # self.image_sub = message_filters.Subscriber(
        #                     "/kinect2/qhd/image_color", Image
        #                 )
        self.image_sub = message_filters.Subscriber(
                            "/camera/rgb/image_color", Image
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
        pts = np.array(object_dict[so], np.int32)
        pts = pts.reshape((-1, 1, 2))
        frame = cv2.polylines(frame,[pts], True, (0,255,0), 3, cv2.LINE_AA)
        frame = self.annotate_frame(object_name=so, dst=pts, viz_frame=frame)
        self.detection_count_alt[so] += 1
        print(f'algo 2: {self.detection_count_alt}')

        info = self.extract_object_info("give me that")
        print(f'object_info: {info}, object: {so}')
        
        
        # eye_line = Line(self.arm_points[[5,3]])
        object_line = Line(np.array([[min_x, y], [max_x, y]]))
        intersection_arm_line = object_line.intersect(arm_line).astype(np.int32)
        # intersection_eye_line = object_line.intersect(eye_line).astype(np.uint16)
        object_distance = np.linalg.norm(object_center - intersection_arm_line, axis=1)
        selected_object = list(object_dict.keys())[np.argmin(object_distance)] 
        self.total_frame += 1
        self.detection_count[selected_object] += 1
        
        image_str = f'Action: {info["action"]}\nObject: {so}'
        image_str = image_str.split('\n')

        pad = 15
        min_x, min_y, max_x = 450, 50, 600
        max_y = min_y+int(pad*(1+len(image_str)))
        frame = cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (55,55,180), 2)
        points = np.array([[min_x, min_y], [min_x, max_y], [max_x, max_y], [max_x, min_y]],
                  dtype=np.int32)

        cv2.fillPoly(frame, [points], (100, 180, 150))
        for i in range(len(image_str)):
            frame = cv2.putText(frame, image_str[i], (min_x+pad, min_y+int(pad*(i+1))), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                                (255, 0, 0), 1, cv2.LINE_AA)

        msg = ros_numpy.msgify(Image, frame, encoding='bgr8')
        self.image_pub.publish(msg)
        
    # def extract_object_info(self, instruction_msg):
        
    #     matcher.add("action", None, [
    #         {"POS": "VERB"},
    #         {"TEXT": "me", "OP": "?"},
    #         {"POS": "DET", "OP": "*"},
    #         {"TEXT": "that", "OP": "*"},
    #         {"POS": "ADJ", "OP": "*"},
    #         {"POS": "NOUN", "OP": "*"}])
        
    #     matcher.add("object", None, [
    #         {"POS": "VERB"},
    #         {"TEXT": "me", "OP": "*"},
    #         {"POS": "DET", "OP": "*"},
    #         {"TEXT": "that", "OP": "*"},
    #         {"POS": "ADJ", "OP": "*"},
    #         {"POS": "NOUN", "OP": "+"}])
    #     matcher.add("pointing-identifier", None, [{"LEMMA": {"IN": ["this", "that"]}}])
    #     matcher.add("attr", None, [{"POS": "ADJ", "OP": "+"}, {"POS": "NOUN"}])
    #     matcher.add("pos", None, [{"LEMMA": {"IN": ["right", "left", "front", "back"]}}])

    #     doc = nlp(instruction_msg)
    #     matches = matcher(doc)
    #     object_info = {}
    #     object_name = ""
    #     action = ""
    #     attr = []
    #     pos = ""
    #     pointing_identifier = False

    #     for match_id, start, end in matches:
    #         string_id = nlp.vocab.strings[match_id]
    #         print(string_id)
    #         if string_id == "action":

    #             action = doc[start]
            
    #         if string_id == "object":
    #             object_name = doc[end-1]
            
    #         if string_id == "attr":
    #             object_name = doc[end-1]
    #             attr.append(doc[start])
            
    #         if string_id == "pointing-identifier":
    #             pointing_identifier = True
            
    #         if string_id == "pos":
    #             pos = doc[start]
    #         span = doc[start:end]  # The matched span
    #         print(string_id, start, end, span.text)

    #     object_info[object_name] = {}
    #     object_info[object_name]["action"] = action
    #     object_info[object_name]["attr"] = attr
    #     object_info[object_name]["pos"] = pos
    #     object_info[object_name]["pointing-identifier"] = pointing_identifier
        
    #     return object_info

    def extract_object_info(self, instruction_msg):
        # matcher.add("action", None, [{"POS": "VERB"},
        #                              {"POS": "PRON", "OP": "*"},
        #                              {},
        #                              {"POS": "ADJ", "OP": "*"},
        #                              {"POS": "NOUN"}])
        matcher.add("action", None, [
            {"POS": "VERB"},
            {"TEXT": "me", "OP": "?"},
            {"POS": "DET", "OP": "*"},
            {"TEXT": "that", "OP": "*"},
            {"POS": "ADJ", "OP": "*"},
            {"POS": "NOUN", "OP": "*"}])
        matcher.add("navigation", None, [{"LEMMA": {"IN": ["go", "come", "move", "turn"]}}])

        matcher.add("attr", None, [{"TAG": "JJ", "OP": "+"}, {"POS": "NOUN"}])
        matcher.add("pos", None, [{"LEMMA": {"IN": ["right", "left", "front", "back"]}}])

        doc = nlp(instruction_msg)
        matches = matcher(doc)
        object_info = {}
        object_name = "None"
        action = "None"
        attr = []
        pos = "None"
        navigation = "None"
        is_navigation = False
        
        for match_id, start, end in matches:
            string_id = nlp.vocab.strings[match_id]
            # print(string_id)
            if string_id == "action":
                object_name = doc[end-1]
                action = doc[start]
            if string_id == "navigation":
                is_navigation = True
                navigation = doc[start]
            if string_id == "attr":
                attr.append(doc[start])

            if string_id == "pos":
                pos = doc[start]

        object_info["no"] = self.counter
        self.counter += 1
        object_info["object"] = object_name
        object_info["action"] = navigation if is_navigation else action
        object_info["attr"] = "None" if len(attr) is 0 else attr
        object_info["pos"] = pos
        return object_info
    
    def annotate_frame(self, viz_frame, dst, object_name):
        # viz_frame = cv2.polylines(
        #                             viz_frame,
        #                             [dst],
        #                             True,
        #                             255,
        #                             1,
        #                             cv2.LINE_AA
        #                         )

        dst = np.squeeze(dst, axis=1)
        tc = (dst[3] + dst[0])/2
        tc = (tc + dst[0])/2

        text_loc = np.array([tc[0], tc[1] - 20], dtype=np.int16)
        base, tangent = dst[3] - dst[0]
        text_angle = np.arctan2(-tangent, base)*180/np.pi
        viz_frame = draw_angled_text(
                            object_name,
                            text_loc,
                            text_angle,
                            viz_frame
                        )
        return viz_frame
    
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
