import random
import time
from threading import Thread

import cv2
import numpy as np
import skimage.morphology as morph
import utils
from scipy.spatial.transform import Rotation

from .cameras import KinectClient, RealSense
from .grippers import WSG50
from .realur5 import UR5RTDE, UR5URX, UR5PairRTDE


def get_largest_component(arr):
    # label connected components for mask
    labeled_arr, num_components = \
        morph.label(
            arr, return_num=True,
            background=0)
    masks = [(i, (labeled_arr == i).astype(np.uint8))
             for i in range(0, num_components)]
    masks.append((
        len(masks),
        1-(np.sum(mask for i, mask in masks) != 0)))
    sorted_volumes = sorted(
        masks, key=lambda item: np.count_nonzero(item[1]),
        reverse=True)
    for i, mask in sorted_volumes:
        if arr[mask == 1].sum() == 0:
            continue
        return mask


class BagEnv():
    def __init__(self, robot=True, realsense=True, robot_home=True, bag_type='rss'):
        self.cam = KinectClient('128.59.23.32', '8080', fielt_bg=False)
        self.cam_intr = self.cam.get_intr()
        self.crop_info = ((140, 430), 400) # [corner, size]
        self.crop_size = [[self.crop_info[0][0], self.crop_info[0][0] + self.crop_info[1]], \
                          [self.crop_info[0][1], self.crop_info[0][1] + self.crop_info[1]]]
        self.translate_mat = utils.translate2d([self.crop_size[0][0], self.crop_size[1][0]]).T
        self.camera2table = np.loadtxt('real_world/cam_pose/cam2table_pose.txt')
        self.camera2left = np.loadtxt('real_world/cam_pose/cam2left_pose.txt')
        self.camera2right = np.loadtxt('real_world/cam_pose/cam2right_pose.txt')
        self.camera2blow = np.loadtxt('real_world/cam_pose/cam2blow_pose.txt')
        self.table2camera = np.linalg.inv(self.camera2table)
        self.table2left = self.camera2left @ self.table2camera
        self.table2right = self.camera2right @ self.table2camera
        self.table2blow = self.camera2blow @ self.table2camera

        self.ee_pos = np.array([
            [-0.15, -0.45, 0.45],
            [0.15, -0.45, 0.45],
        ])
        self.blow_pose = (np.array([0, -0.75, 0.75]), np.array([0, 0, -np.pi/4]))
        if robot:
            self.ip_port = [
                ('192.168.0.142', 30002, 30003),    # left ur5 (with wsg50)
                ('192.168.0.139', 30002, 30003),    # right ur5 (with RG2)
                ('192.168.0.204', 30002, 30003),    # blow ur5 (with blower)
                ('192.168.0.231', 1001)             # wsg50
            ]
            # self.wsg50 = None
            self.wsg50 = WSG50(self.ip_port[3][0], self.ip_port[3][1])
            self.blow_ur5 = UR5URX(ip=self.ip_port[0][0], home_joint=np.array([-90, -90, 0, 0, 90, 90]) / 180 * np.pi, gripper='blower')
            
            self.left_ur5 = UR5RTDE(self.ip_port[1][0], self.wsg50)
            self.right_ur5 = UR5RTDE(self.ip_port[2][0], 'rg2')
            self.ur5_pair = UR5PairRTDE(self.left_ur5, self.right_ur5)
            if robot_home:
                self.ur5_pair.home()
                self.blow_ur5.home()
            
            self.move_blower(self.blow_pose[0], self.blow_pose[1])
        
        self.labeling = False
        if realsense:
            self.label_frame_gap = 0.05
            self.realsense = RealSense('0.0.0.0', 50011)
            self.label_images = list()
            self.all_label_images = [list(), list()]
            self.labeling_damon = Thread(target=self.label_fn, daemon=True)
            self.labeling_damon.start()
        else:
            self.label_images = None

        self.bag_info_dict = {
            'rss':{
                'x_range': [0.33, 0.38, 0.38],
                'area_threshold': 34000,
            },
            'white':{
                'x_range': [0.31, 0.36, 0.37],
                'area_threshold': 35000,
            },
            'yellow':{
                'x_range': [0.33, 0.38, 0.41],
                'area_threshold': 45000,
            },
            'blue':{
                'x_range': [0.43, 0.48, 0.48],
                # 'area_threshold': 46500,
                'area_threshold': 35000,
            }
        }
        self.bag_type = bag_type
        self.x_range = self.bag_info_dict[bag_type]['x_range']
        self.area_threshold = self.bag_info_dict[bag_type]['area_threshold']

        self.grasping_space = np.array([
            [-0.55, -0.35],     # y-axis
            [0.38, 0.52],       # z-axis
            [0, np.pi/6]        # tilt
        ])
        self.blow_space = np.array([
            [-0.85, -0.65],                 # y-axis
            [0.7, 0.8],                     # z-axis
            [-75/180*np.pi, -15/180*np.pi]  # orientation
        ])

    def move_to(self, wrd_p1=None, wrd_p2=None, tilt=0, middle_left=None, middle_right=None, speed=0.5, acceleration=0.3):
        wrd_p1 = wrd_p1 if wrd_p1 is not None else self.ee_pos[0]
        wrd_p2 = wrd_p2 if wrd_p2 is not None else self.ee_pos[1]
        p_left = utils.apply_transformation(wrd_p1, self.table2left)
        p_right = utils.apply_transformation(wrd_p2, self.table2right)

        left_rotvec = (Rotation.from_euler('x', tilt) * Rotation.from_euler('y', -np.pi/6) * Rotation.from_euler('z', 0) * Rotation.from_rotvec([np.pi, 0, 0])).as_rotvec()
        right_rotvec = (Rotation.from_euler('x', tilt) * Rotation.from_euler('y', np.pi/6) * Rotation.from_euler('z', -np.pi/2) * Rotation.from_rotvec([np.pi, 0, 0])).as_rotvec()
        
        p_left = np.concatenate([p_left, left_rotvec]).tolist()
        p_right = np.concatenate([p_right, right_rotvec]).tolist()

        if middle_left is not None:
            middle_left_rotvec = left_rotvec
            middle_left = np.concatenate([utils.apply_transformation(middle_left, self.table2left), middle_left_rotvec]).tolist()
            p_left = [middle_left, p_left]
        if middle_right is not None:
            middle_right_rotvec =  right_rotvec
            middle_right = np.concatenate([utils.apply_transformation(middle_right, self.table2right), middle_right_rotvec]).tolist()
            p_right = [middle_right, p_right]

        self.ur5_pair.movel(p_left, p_right, speed, acceleration)

    def shake(self, tilt, speed=3, acceleration=1.5, num_iter=2):
        p_left_list, p_right_list = list(), list()
        for _ in range(num_iter):
            p_left = np.array([self.ee_pos[0][0], self.grasping_space[0][0], self.ee_pos[0][2]])
            p_right = np.array([self.ee_pos[1][0], self.grasping_space[0][0], self.ee_pos[1][2]])
            p_left_list.append(p_left)
            p_right_list.append(p_right)
            p_left = np.array([self.ee_pos[0][0], self.grasping_space[0][1], self.ee_pos[0][2]])
            p_right = np.array([self.ee_pos[1][0], self.grasping_space[0][1], self.ee_pos[1][2]])
            p_left_list.append(p_left)
            p_right_list.append(p_right)
        p_left_list.append(self.ee_pos[0])
        p_right_list.append(self.ee_pos[1])

        pose_left_list, pose_right_list = list(), list()
        for p_left in p_left_list:
            p_left = utils.apply_transformation(p_left, self.table2left)
            left_rotvec = (Rotation.from_euler('x', tilt) * Rotation.from_euler('y', -np.pi/6) * Rotation.from_euler('z', 0) * Rotation.from_rotvec([np.pi, 0, 0])).as_rotvec()
            p_left = np.concatenate([p_left, left_rotvec]).tolist()
            pose_left_list.append(p_left)

        for p_right in p_right_list:
            p_right = utils.apply_transformation(p_right, self.table2right)
            right_rotvec = (Rotation.from_euler('x', tilt) * Rotation.from_euler('y', np.pi/6) * Rotation.from_euler('z', -np.pi/2) * Rotation.from_rotvec([np.pi, 0, 0])).as_rotvec()
            p_right = np.concatenate([p_right, right_rotvec]).tolist()
            pose_right_list.append(p_right)
        
        self.ur5_pair.movel(pose_left_list, pose_right_list, speed, acceleration)

    def move_blower(self, position=None, orientation=None, delta_action=None, speed=0.5, acceleration=0.3):
        if delta_action is not None:
            position = self.blow_pose[0] + np.array([0, delta_action[0], delta_action[1]])
            orientation = self.blow_pose[1] + np.array([0, 0, delta_action[2]])
        self.blow_pose = (np.array(position), np.array(orientation))
        position = utils.apply_transformation(np.array(position), self.table2blow)
        orientation = (Rotation.from_euler('x', orientation[2]) * Rotation.from_euler('z', -np.pi/2)).as_rotvec()
        pose = np.concatenate([position, orientation])
        self.blow_ur5.movel(pose, speed=speed, acceleration=acceleration)


    def move_gripper(self, position, distance, tilt):
        self.ee_pos[0][1] = self.ee_pos[1][1] = position[0]
        self.ee_pos[0][2] = self.ee_pos[1][2] = position[1]
        self.ee_pos[0][0] = -distance / 2
        self.ee_pos[1][0] = distance / 2
        self.move_to(tilt=tilt)


    def get_random_grasp_position(self, hard=False):
        if hard:
            distance = random.uniform(self.x_range[0], self.x_range[1]) - 0.03
            tilt = random.uniform(self.grasping_space[2][0], self.grasping_space[2][1]) - np.pi/6
        else:
            distance = random.uniform(self.x_range[0], self.x_range[1])
            tilt = random.uniform(self.grasping_space[2][0], self.grasping_space[2][1])
        position = np.array([
            random.uniform(self.grasping_space[0][0], self.grasping_space[0][1]),
            random.uniform(self.grasping_space[1][0], self.grasping_space[1][1])
        ])
        return position, distance, tilt


    def set_random_grasping(self, test_blower=True):
        succ = False
        for _ in range(5):
            self.blow_ur5.close_gripper()
            last_position = np.array([self.ee_pos[0][1], self.ee_pos[0][2]])
            
            while True:
                new_position, distance, tilt = self.get_random_grasp_position()
                if np.linalg.norm(last_position - new_position) < 0.08:
                    break

            self.move_gripper(new_position, distance, tilt)

            if not test_blower:
                succ = True
                break
            self.blow_ur5.open_gripper()

            if not self.get_reward()[0]:
                succ = True
                break
        if not succ:
            print('[Error] fail to set random grasping')
        return new_position, distance, tilt


    def get_current_action(self):
        blow_position = self.blow_pose[0]
        blow_orientation = self.blow_pose[1]
        return np.array([blow_position[1], blow_position[2], blow_orientation[2]])


    def get_random_delta_actions(self, action_num=64):
        blow_position = self.blow_pose[0]
        blow_orientation = self.blow_pose[1]
        y_min = max(self.blow_space[0][0], blow_position[1] - 0.05)
        y_max = min(self.blow_space[0][1], blow_position[1] + 0.05)
        z_min = max(self.blow_space[1][0], blow_position[2] - 0.05)
        z_max = min(self.blow_space[1][1], blow_position[2] + 0.05)
        theta_min = max(self.blow_space[2][0], blow_orientation[2] - np.pi/6)
        theta_max = min(self.blow_space[2][1], blow_orientation[2] + np.pi/6)

        delta_actions = list()
        for _ in range(action_num):
            delta_actions.append(np.array([
                random.uniform(y_min, y_max),
                random.uniform(z_min, z_max),
                random.uniform(theta_min, theta_max),
            ]) - np.array([blow_position[1], blow_position[2], blow_orientation[2]]))
        return delta_actions

    def get_random_blow_position(self, angle=False, low=False):
        if angle:
            return np.array([
                random.uniform(self.blow_space[0][0], self.blow_space[0][1]),   # y-axis
                random.uniform(self.blow_space[1][0], self.blow_space[1][1]),   # z-axis
                random.uniform(self.blow_space[2][0], self.blow_space[2][1])
            ])
        else:
            if low:
                return np.array([
                    random.uniform(self.blow_space[0][0], self.blow_space[0][1]-0.1),   # y-axis
                    random.uniform(self.blow_space[1][0], self.blow_space[1][1]-0.1)    # z-axis
                ])
            else:
                return np.array([
                    random.uniform(self.blow_space[0][0], self.blow_space[0][1]),   # y-axis
                    random.uniform(self.blow_space[1][0], self.blow_space[1][1])    # z-axis
                ])

    def get_area(self, color_img, depth_img):
        # TODO: change this function to maskout the bag
        hsv = cv2.cvtColor(color_img, cv2.COLOR_RGB2HSV)
        if self.bag_type in['blue', 'yellow']:
            mask = np.logical_and(hsv[:, :, 1] > 60, hsv[:, :, 2] > 60)
        elif self.bag_type in ['rss', 'white']:
            mask = hsv[:, :, 2] > 190
        else:
            raise NotImplementedError()
        mask = mask[125:, 520:980]
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4)), iterations=4)
        mask = get_largest_component(mask)
        area = np.sum(mask)
        return area

    def get_reward(self):
        time.sleep(2)
        self.labeling = True
        time.sleep(2)
        self.labeling = False
        raw_image_list, area_list = list(), list()
        for color_img, depth_img in self.label_images:
            raw_image_list.append(color_img)
            area = self.get_area(color_img, depth_img)
            area_list.append(area)

        self.label_images = list()
        avg_area = np.mean(area_list)
        return avg_area > self.area_threshold, area_list, raw_image_list


    def get_observation(self):
        self.color_img, self.depth_img = self.cam.get_camera_data(n=1)
        return self.color_img, self.depth_img


    def terminate(self):
        self.left_ur5.rtde_c.disconnect()
        self.right_ur5.rtde_c.disconnect()
        self.blow_ur5.robot.secmon.close()


    def label_fn(self):
        while True:
            init_time = time.time()
            color_img, depth_img = self.realsense.get_camera_data()
            if self.labeling:
                self.label_images.append((color_img, depth_img))
            sleep_time = self.label_frame_gap - (time.time() - init_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
