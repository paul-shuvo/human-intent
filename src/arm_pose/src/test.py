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


class CanvasInteraction:
    def __init__(self, canvas_box):  
        """
        Initializes the class and starts the subscribers.

        Args:
            canvas_pts (`ndarray`): Three pixel locations (co-planar in 3D space) on an image that describe a surface.
        """
        # self.K is the camera matrix retrieved from /kinect2/qhd/camera_info 
        self.K = np.array([540.68603515625, 0.0, 479.75, 0.0, 540.68603515625, 269.75, 0.0, 0.0, 1.0]).reshape((3,3), order='C')
        self.canvas_box = canvas_box
        self.canvas_pts_3D = np.zeros((3, 3))
        self.arm_points = np.zeros((2, 2), dtype=np.int16).tolist()
        self.arm_points_3D = np.zeros((2, 3))
        self.canvas_plane = None
        self.compute_canvas_once = True
        self.arm_pose= PoseStamped()

        
        rospy.init_node('projection_node_2', anonymous=False)
        # self.pub = rospy.Publisher('/point_on_canvas', PoseStamped, queue_size=10)
        # self.msg = PoseStamped()
        self.frame_id = 'kinect2_rgb_optical_frame'
        # forearm_pose_sub = message_filters.Subscriber('/kinect2/qhd/camera_info', CameraInfo)
        self.forearm_pose1_pub = rospy.Publisher('/forearm_pose_1', PoseStamped, queue_size=10)
        self.forearm_pose2_pub = rospy.Publisher('/forearm_pose_2', PoseStamped, queue_size=10)
        self.msg1 = PoseStamped()
        self.msg2 = PoseStamped()

        forearm_pose_sub = message_filters.Subscriber('/forearm_pose', Floats)
        pointcloud_sub = message_filters.Subscriber('/kinect2/qhd/points', PointCloud2)      
        ts = message_filters.ApproximateTimeSynchronizer([forearm_pose_sub, pointcloud_sub], 10, 1, allow_headerless=True) # Changed code
        ts.registerCallback(self.callback)
        # spin
        rospy.spin()
    
    def pose_from_vector3D(self, position, waypoint):
        #http://lolengine.net/blog/2013/09/18/beautiful-maths-quaternion-from-vectors
        pass
            
    def update_points_arm_3d(self, forearm_pose, pointcloud, arm='left'):
        # choose left arm for now
        arm_loc_np = np.asarray(forearm_pose.data, dtype=np.int16)
        if arm is 'right':
            arm_joint_pts = [3, 1]
        else:
            arm_joint_pts = [0, 2] # 2 is LElbow, 0 is LWrist
        # right_arm_joint_pts = [0, 2]
        self.arm_points = arm_loc_np.reshape((arm_loc_np.shape[0]//2, -1), order='C')[arm_joint_pts]
        arb_arm_points = segment_arb_pts(self.arm_points, n_pts=10)
        # print(self.arm_points)
        # hold the current values
        # uv = arb_arm_points.astype(int).tolist()
        # print(uv)
        # count = 0
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
        
        # if self.compute_canvas_once:
        #     self.points_3d_on_plane(self.canvas_box, pointcloud)
        #     self.compute_canvas_once = False
        # else:
        #     self.points_3d_on_plane(self.canvas_box, pointcloud)
            
        self.update_points_arm_3d(forearm_pose, pointcloud)
        # self.pose_from_vector3D(self.arm_points_3D[0], self.arm_points_3D[1] - self.arm_points_3D[0])
        
        try:
            # left_arm_line_3D = Line3D(Point3D(*self.arm_points_3D[0, :]), Point3D(*self.arm_points_3D[1, :]))
            # self.canvas_plane = Plane(Point3D(*self.canvas_pts_3D[0, :]), Point3D(*self.canvas_pts_3D[1, :]), Point3D(*self.canvas_pts_3D[2, :]))
            # canvas_plane_point = self.canvas_plane.intersection(left_arm_line_3D)[0].evalf()
            
            self.msg1.header.stamp = rospy.Time.now()
            self.msg1.header.frame_id = 'kinect2_rgb_optical_frame'
            self.msg1.pose.position.x = self.arm_points_3D[0, 0]
            self.msg1.pose.position.y = self.arm_points_3D[0, 1]
            self.msg1.pose.position.z = self.arm_points_3D[0, 2]
            # Make sure the quaternion is valid and normalized
            self.msg1.pose.orientation.x = 0.0
            self.msg1.pose.orientation.y = 0.0
            self.msg1.pose.orientation.z = 0.0
            self.msg1.pose.orientation.w = 1.0
            
            self.msg2.header.stamp = rospy.Time.now()
            self.msg2.header.frame_id = 'kinect2_rgb_optical_frame'
            self.msg2.pose.position.x = self.arm_points_3D[1, 0]
            self.msg2.pose.position.y = self.arm_points_3D[1, 1]
            self.msg2.pose.position.z = self.arm_points_3D[1, 2]
            # Make sure the quaternion is valid and normalized
            self.msg2.pose.orientation.x = 0.0
            self.msg2.pose.orientation.y = 0.0
            self.msg2.pose.orientation.z = 0.0
            self.msg2.pose.orientation.w = 1.0
            # # self.frame_id += 1
            # print(canvas_plane_point)
            # print(canvas_plane_point.y)
        except ValueError as error:
            rospy.loginfo(error)

        self.forearm_pose1_pub.publish(self.msg1)
        self.forearm_pose2_pub.publish(self.msg2)
        print('msg published')



#     # try:
#     #     listen()
#     # except rospy.ROSInterruptException as error:
#     #     rospy.loginfo(error)
#     # 
#     # 
# # #!/usr/bin/env python
# import rospy
# from arm_pose.msg import Floats
# # import ros_np_multiarray as rnm
# import numpy as np
# from rospy.numpy_msg import numpy_msg
# from image_geometry import PinholeCameraModel
# import message_filters
# from sensor_msgs.msg import Image, CameraInfo
# from geometry_msgs.msg import PoseStamped

import cv2
import os

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

        # self.count = 0
        # init the node
        rospy.init_node('show_image_node_test', anonymous=False)
  

        # self.image_sub = message_filters.Subscriber('/kinect2/qhd/image_color_rect', Image)
        # self.forearm_pose_sub = message_filters.Subscriber('/forearm_pose', Floats)
        r = rospy.Rate(2)

        while not rospy.is_shutdown():
            rospy.Subscriber("/kinect2/qhd/image_color_rect", Image, self.callback)
            r.sleep()

    def callback(self, image):
        # if selected_point_sub is not None:
        # self.camera_model.fromCameraInfo(camera_info)
    
        frame = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, -1)
    

        cv2.imshow('image', frame)
        cv2.waitKey(30)
        


# def main():
#     """ main function
#     """
#     node = ShowImage()

if __name__ == '__main__':

    # print(camera_info)
    # ShowImage()
    
# if __name__ == '__main__':
    canvas_pts=np.array([[600, 300], [600, 400], [800, 400], [800, 300]], dtype=np.int32)
    CanvasInteraction(canvas_box=canvas_pts)

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

   





