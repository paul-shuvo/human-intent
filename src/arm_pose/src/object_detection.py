#!/usr/bin/env python
from os.path import dirname, abspath, join
from os import getcwd, listdir, chdir
import numpy as np
import time
import cv2
import json

# Change directory to the file directory
dir_path = dirname(abspath(__file__))
chdir(dir_path)

# ROS imports

import rospy
from std_msgs.msg import String
from arm_pose.msg import Floats
from sensor_msgs.msg import PointCloud2, Image
# from rospy.numpy_msg import numpy_msg

# TODO: Add docstring
# TODO: Make an additional pose detection class

class ObjectDetection():
    def __init__(self, objects='all'):
        # '/objects' contains the images of the object
        self.object_path = join(dir_path, 'objects')
        # print(self.object_path)
        image_files = [join(self.object_path, f) for f in listdir(self.object_path) if f.endswith(('.jpg', '.png'))]
        self.query_object_im = {}
        # TODO: finish teh else logic
        if objects is 'all':
            for im_file in image_files:
                object_name = im_file.split('/')[-1].split('.')[0]
                # print(object_name)
                try:
                    self.query_object_im[object_name] = cv2.imread(im_file)
                    print(self.query_object_im[object_name].dtype)
                    # break
                except:
                    rospy.loginfo(f'Image couldn\'t be red at: \n {im_file}')
        # # print(image_files)
        # cv2.imshow('image', self.query_object_im['book'])
        # cv2.waitKey(5000)
        self.frame_rate = 2
        self.prev = 0

        rospy.init_node('object_detection', anonymous=False)
        self.obj_boundary_msg = ''
        self.obj_boundary_info = {}
        self.obj_boundary_pub = rospy.Publisher('/detected_object', String, queue_size=10)

        r = rospy.Rate(self.frame_rate) # 10Hz

        while not rospy.is_shutdown():
            rospy.Subscriber("/kinect2/qhd/image_color_rect", Image, self.callback)
            r.sleep()
    
    def callback(self, kinect_image):
        image = np.frombuffer(kinect_image.data, dtype=np.uint8).reshape(kinect_image.height, kinect_image.width, -1)
        if image is None:
            rospy.loginfo('invalid image received')
            return
        
        time_elapsed = time.time() - self.prev
        if time_elapsed > 1. / self.frame_rate:
            self.prev = time.time()
            for object_name, query_im in self.query_object_im.items():
                self.detect(object_name, query_im, image)

        # Convert the dictionary to string
        self.obj_boundary_msg = json.dumps(self.obj_boundary_info)
        self.obj_boundary_pub.publish(self.obj_boundary_msg)

    def detect(self, object_name, query_im, kinect_im, show_image=False):
        # minimum matching points needed to consider a match
        MIN_MATCH_COUNT = 10
        
        # img1 = cv.imread('box.png',0)          # queryImage
        # img2 = cv.imread('box_in_scene.png',0) # trainImage

        # Initiate SIFT detector
        sift = cv2.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(query_im,None)
        kp2, des2 = sift.detectAndCompute(kinect_im,None)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)

        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()
            h,w,d = query_im.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts,M)
            # update the location of the object in the image
            # converted to list as ndarray object is not json serializable
            self.obj_boundary_info[object_name] = dst[:, 0, :].tolist()
            if show_image:
                result = cv2.polylines(kinect_im,[np.int32(dst)],True,255,3, cv2.LINE_AA)
                cv2.imshow('Detected Objects', result)
                cv2.waitKey(30)
        else:
            # Set None if the object isn't detected
            self.obj_boundary_info[object_name] = None
            rospy.loginfo( "Not enough matches are found - {}/{}".
                format(len(good), MIN_MATCH_COUNT) )
            matchesMask = None
        

if __name__ == "__main__":
    ObjectDetection()