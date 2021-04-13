#!/usr/bin/env python
from sklearn.utils.validation import _num_samples
import rospy
from arm_pose.msg import Floats

# import ros_np_multiarray as rnm
import numpy as np
import json

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
        pointcloud_sub = message_filters.Subscriber("/camera/depth_registered/points", PointCloud2)
        detected_object_sub = message_filters.Subscriber("/detected_object", String)

        ts = message_filters.ApproximateTimeSynchronizer(
            [forearm_pose_sub, pointcloud_sub, detected_object_sub],
            10,
            1,
            allow_headerless=True,
        )  # Changed code
        ts.registerCallback(self.callback)
        # spin
        rospy.spin()

    def callback(self, forearm_pose, pointcloud, detected_object):
        # if self.compute_once:
        # print('here')
        self.update_arm_points(forearm_pose, pointcloud)
        if self.compute_once[0]:
            self.update_plane(pointcloud, detected_object)
            self.compute_once[0] = False
        elif not self.compute_once[0] and not self.compute_once[1]:
            self.update_plane(pointcloud, detected_object)

        # print(self.object_center)
        object_distance = {}
        self.pose_array.header.frame_id = self.frame_id
        pose_msg_arr = []
        for object_name, plane in self.detected_object_plane.items():
            # print(object_name)
            intersect_point = plane.intersection(Line3D(self.arm_points_3D))
            object_distance[object_name] = np.linalg.norm(intersect_point[0, 0, :] - self.object_center[object_name])
            # print(intersect_point)
            pose_msg = Pose()
            # pose_msg.header.stamp = rospy.Time.now()
            # pose_msg.header.frame_id = self.frame_id
            pose_msg.position.x = intersect_point[0,0,0]
            pose_msg.position.y = intersect_point[0,0,1]
            pose_msg.position.z = intersect_point[0,0,2]
            # Make sure the quaternion is valid and normalized
            pose_msg.orientation.x = 0.0
            pose_msg.orientation.y = 0.0
            pose_msg.orientation.z = 0.0
            pose_msg.orientation.w = 1.0

            pose_msg_arr.append(pose_msg)

        self.pose_array.poses = pose_msg_arr
        self.pose_array.header.stamp = rospy.Time.now()

        self.pose_array_pub.publish(self.pose_array)
        # self.pub.publish(self.msg)
        # print(self.msg)
        min_ = min(object_distance.items(), key=lambda x: x[1])
        # self.detection_count[min_[0]] += 1
        # self.detection_voting[min_[0]] += 1
        # self.voting_interval += 1
        # if self.voting_interval == 30:
        #     max_ = max(self.detection_voting.items(), key=lambda x: x[1])
        #     self.detection_voting_count[max_[0]] += 1
        #     print('detection count: ', self.detection_count)
        #     print('detection voting: ', self.detection_voting)
        #     print('detection voting count: ', self.detection_voting_count)
        #     print('--------------------------------------')
        #     self.voting_interval = 0
        #     self.detection_voting = {
        #         'cheeze-it': 0,
        #         'book-2': 0,
        #         'book-3': 0
        #     } 
        # print(f'minimum is: {min_}')
        print(min_[0])
        # print(object_distance)
        # print('--------------------------------------')
        self.counter += 1

    def update_arm_points(self, forearm_pose, pointcloud):
        # choose left arm for now
        arm_loc_np = np.asarray(forearm_pose.data, dtype=np.int16)
        left_arm_joint_pts = [0, 2]
        right_arm_joint_pts = [1, 3]

        self.arm_points = arm_loc_np.reshape((arm_loc_np.shape[0] // 2, -1), order="C")[
            right_arm_joint_pts
        ].tolist()

        pre_arm_points_3D = self.arm_points_3D
        count = 0
        for dt in pc2.read_points(
            pointcloud,
            field_names={"x", "y", "z"},
            skip_nans=False,
            uvs=self.arm_points,
        ):
            # check if it's not NaN
            if dt[0] == dt[0]:
                # print(f'x: {dt[0]}, y: {dt[1]}, z: {dt[2]}')
                self.arm_points_3D[count, :] = dt
            else:
                self.arm_points_3D = pre_arm_points_3D
                break
            count += 1

            # print(self.arm_points_3D)
        # print(self.arm_points_3D)
        # pre_detected_object_plane = self.detected_object_plane
        # t_val = 0.75
        # if erros using the json.load() check the following link
        # https://stackoverflow.com/questions/11174024/attributeerrorstr-object-has-no-attribute-read

    def update_plane(self, pointcloud, detected_object):

        if self.regress_plane:
            # if self.compute_once:
            for object_name, bounding_box in json.loads(detected_object.data).items():
                if bounding_box is None:
                    continue
                is_valid, points_3d = self.get_points_on_plane(object_name, bounding_box, pointcloud)
                # print(is_valid)
                if not is_valid:
                    continue
                from sklearn.linear_model import RANSACRegressor
                import itertools

                comb = np.array(list(itertools.combinations(points_3d, r=3)))
                p = comb[:1]
                n = Plane(comb).n
                comb = comb.reshape((comb.shape[0], -1))
                reg = RANSACRegressor(random_state=0).fit(comb, n)

                self.detected_object_plane[object_name] = Plane(p, n=reg.predict(p.reshape((1, -1))))
            # self.compute_once = False
            # print('computing')
            # else:
            #     for object_name, bounding_box in json.loads(detected_object.data).items():
            #         if bounding_box == None:
            #             continue
            #         is_valid, points_3d = self.get_points_on_plane(bounding_box, pointcloud)
            #         # print(is_valid)
            #         if not is_valid:
            #             continue
            #         from sklearn.linear_model import RANSACRegressor
            #         import itertools

            #         comb = np.array(list(itertools.combinations(points_3d, r=3)))
            #         p = comb[:1]
            #         n = Plane(comb).n
            #         comb = comb.reshape((comb.shape[0], -1))
            #         reg = RANSACRegressor(random_state=0).fit(comb, n)

            #         self.detected_object_plane[object_name] = Plane(p, n=reg.predict(p.reshape((1, -1))))
            #     print('computing')
        else:
            for object_name, bounding_box in json.loads(detected_object.data).items():
                is_valid, points_3d = self.get_points_on_plane(object_name, bounding_box, pointcloud)
                if is_valid:
                    self.detected_object_plane[object_name] = Plane(points_3d)
            # self.compute_once = False
            print('computing')


        # print(self.detected_object_plane)
            # except TypeError as e:
            #     rospy.loginfo(e)
            #     continue
        # print('end')
    
    def best_fit(reg, X, y):
        pass

    def get_points_on_plane(self, object_name, bounding_box, pointcloud):
        if bounding_box is None:
            return False, False
        # print(bounding_box)
        n_points = 30
        bounding_box = np.array(bounding_box)
        points = (
            np.vstack(
                (
                    self.points_within_triangle(bounding_box[:3], n_points),
                    self.points_within_triangle(bounding_box[1:], n_points),
                )
            )
            .astype(int)
            .tolist()
        )

        #######################################
        # new code, need to be tested
        sampled_points = self.points_within_circle(bounding_box)
        for dt in pc2.read_points(
                pointcloud, field_names={"x", "y", "z"}, skip_nans=True, uvs=sampled_points
            ):
            self.object_center[object_name] = dt
            break
        #######################################

        if self.regress_plane:
            pt_list = []
            for dt in pc2.read_points(
                pointcloud, field_names={"x", "y", "z"}, skip_nans=True, uvs=points
            ):
                pt_list.append(dt)

            return True, pt_list
        else:
            count = 0
            points_3d = np.zeros((3, 3))
            for dt in pc2.read_points(
                pointcloud, field_names={"x", "y", "z"}, skip_nans=True, uvs=points
            ):
                # check if it's not NaN
                # if dt[0] == dt[0]:
                # print(f'x: {dt[0]}, y: {dt[1]}, z: {dt[2]}')
                points_3d[count] = dt
                count += 1
                if count == 3:
                    break

            return [True, points_3d] if count is 3 else [False, None]

    def points_within_triangle(self, v, n):
        """
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
        """
        x = np.sort(np.random.rand(2, n), axis=0)
        return np.column_stack([x[0], x[1] - x[0], 1.0 - x[1]]) @ v

    def points_within_circle(self, bounding_box):
        _num_samples = 20
        # radius of the circle
        circle_r = 10
        # center of the circle (x, y)
        # box = np.random.random((4,2)) * 50
        center = (bounding_box[0] + bounding_box[3]) / 2
        # center = np.array([20,10])

        # random angle
        alpha = 2 * np.pi * np.random.random((_num_samples, 1))
        # random radius
        r = circle_r * np.sqrt(np.random.random())
        # calculating coordinates
        x = r * np.cos(alpha) + center[0]
        y = r * np.sin(alpha) + center[1]
        x[0] = center[0]
        y[0] = center[1]
        return np.hstack((x, y)).astype(np.int32).tolist()
        # print("Random point", np.hstack((x, y)))

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
