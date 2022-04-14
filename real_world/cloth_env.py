import time
from threading import Thread

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.morphology as morph
import utils
from scipy import ndimage
from scipy.spatial.transform import Rotation

from .cameras import KinectClient, RealSense
from .grippers import WSG50
from .realur5 import UR5RTDE, UR5URX, UR5PairRTDE

# TODO: tune these two parameters!
GRIPPER_LINE = 510
CLOTH_LINE = 570

FOREGROUND_BACKGROUND_DIST = 1.2

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

def is_cloth_grasped(depth, debug=False):
    cloth_mask = cv2.morphologyEx(
        np.logical_and(
            depth < FOREGROUND_BACKGROUND_DIST, depth != 0).astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=4)

    gripper_strip = cloth_mask[GRIPPER_LINE, :]

    if debug:
        cloth_mask = cloth_mask.astype(float)
        cloth_mask[GRIPPER_LINE - 5:GRIPPER_LINE+5, :] = 3
        cloth_mask[CLOTH_LINE - 5:CLOTH_LINE+5, :] = 0.5
        cloth_mask[:, 1280//2] = 0.8
        plt.imshow(cloth_mask)
        plt.show()
        return

    # find grippers
    center = len(gripper_strip)//2
    right_gripper_pix = center + 1
    while not gripper_strip[right_gripper_pix]:
        right_gripper_pix += 1
        if right_gripper_pix == len(gripper_strip) - 1:
            break
    left_gripper_pix = center - 1
    while not gripper_strip[left_gripper_pix]:
        left_gripper_pix -= 1
        if left_gripper_pix == 0:
            break
    center = int((left_gripper_pix + right_gripper_pix)/2)
    cloth_mask[:, :max(left_gripper_pix-100, 1)] = 0
    cloth_mask[:, min(right_gripper_pix+100, cloth_mask.shape[1]):] = 0
    left_grasped = cloth_mask[CLOTH_LINE, :center].sum() > 0
    right_grasped = cloth_mask[CLOTH_LINE, center:].sum() > 0
    return [left_grasped, right_grasped]

def plt_batch(imgs, title=''):
    fig, axes = plt.subplots(3, 3)
    fig.set_figheight(6)
    fig.set_figwidth(7)
    fig.suptitle(title)
    for ax, (img, title) in zip(axes.flatten(), imgs):
        ax.set_title(title)
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def is_cloth_stretched(rgb, depth, angle_tolerance=20, threshold=16, debug=False):
    imshows = []
    fgbg = np.logical_and(depth < FOREGROUND_BACKGROUND_DIST, depth != 0).astype(np.uint8)
    if debug:
        imshows = [(rgb, 'rgb'), (depth, 'depth'), (fgbg.copy(), 'fgbg')]
    fgbg = cv2.morphologyEx(
        fgbg, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=4)
    if debug:
        # fgbg[GRIPPER_LINE - 3:GRIPPER_LINE + 3, :] = 0
        imshows.append((fgbg.copy(), 'mask'))
    gripper_strip = fgbg[GRIPPER_LINE, :]
    # find grippers
    center = len(gripper_strip)//2
    right_gripper_pix = center + 1
    while not gripper_strip[right_gripper_pix]:
        right_gripper_pix += 1
        if right_gripper_pix == len(gripper_strip) - 1:
            break
    left_gripper_pix = center - 1
    while not gripper_strip[left_gripper_pix]:
        left_gripper_pix -= 1
        if left_gripper_pix == 0:
            break
    center = int((left_gripper_pix + right_gripper_pix)/2)
    fgbg[:, :left_gripper_pix+3] = 0
    fgbg[:, right_gripper_pix-3:] = 0
    fgbg[:GRIPPER_LINE, :] = 0

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (5, 5))
    line_mask = cv2.morphologyEx(
        fgbg.copy(), cv2.MORPH_CLOSE, kernel,
        iterations=4)
    
    kernel = np.array([[-1], [0], [1]]*3)
    line_mask = cv2.filter2D(fgbg, -1, kernel)
    if debug:
        imshows.append((fgbg.copy(), 'filtered'))
        imshows.append((line_mask.copy(), 'horizontal edges'))

    line_mask = get_largest_component(
        cv2.morphologyEx(
            line_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (10, 10)), iterations=5))
    if debug:
        plt_batch(imshows)
    
    if debug:
        imshows.append((line_mask.copy(), 'largest component'))

    if line_mask is None:
        return None

    # find angle to rotate to
    points = np.array(np.where(line_mask.copy() == 1)).T
    points = np.array(sorted(points, key=lambda x: x[1]))
    max_x = points[-1][1]
    min_x = points[0][1]
    min_x_y = min(points[(points[:, 1] == min_x)],
                  key=lambda pnt: pnt[0])[0]
    max_x_y = min(points[(points[:, 1] == max_x)],
                  key=lambda pnt: pnt[0])[0]
    angle = 180 * np.arctan((max_x_y-min_x_y)/(max_x - min_x))/np.pi
    line_mask = ndimage.rotate(line_mask, angle, reshape=False)
    if debug:
        print('angle:', angle)
        img = np.zeros(line_mask.shape).astype(np.uint8)
        img = cv2.circle(
            img=img,
            center=(min_x, min_x_y),
            radius=10, color=1, thickness=3)
        img = cv2.circle(
            img=img,
            center=(max_x, max_x_y),
            radius=10, color=1, thickness=3)
        imshows.append((img, 'circled'))
        imshows.append((line_mask.copy(), f'rotated ({angle:.02f}°)'))
    # if angle is too sharp, cloth is probably not stretched
    y_values = np.array(np.where(line_mask == 1))[0, :]
    min_coord = y_values.min()
    max_coord = y_values.max()
    stretchedness = 1/((max_coord - min_coord)/line_mask.shape[0])
    too_tilted = np.abs(angle) > angle_tolerance
    stretch = (not too_tilted) and (stretchedness > threshold)
    print(stretch, 'stretchedness = ', stretchedness, 'threshold = ', threshold)
    if debug:
        print(stretchedness)
        plt_batch(imshows, f'Stretchness: {stretchedness:.02f}, Stretched: {stretch}')
    return stretch

    
class RealWorldEnv():
    def __init__(self, robot=True, realsense=True, robot_home=True, primitive='blow'):
        self.cam = KinectClient('128.59.23.32', '8080', fielt_bg=True)
        self.cam_intr = self.cam.get_intr()
        self.crop_info = ((170, 610), 440) # [corner, size]
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
        if realsense:
            self.realsense = RealSense('0.0.0.0', 50014)

        self.ee_pos = None
        if robot:
            self.ip_port = [
                ('192.168.0.142', 30002, 30003),    # left ur5 (with wsg50)
                ('192.168.0.139', 30002, 30003),    # right ur5 (with RG2)
                ('192.168.0.204', 30002, 30003),    # blow ur5 (with blower)
                ('192.168.0.231', 1001)             # wsg50
            ]
            self.wsg50 = WSG50(self.ip_port[3][0], self.ip_port[3][1])
            self.blow_ur5 = UR5URX(ip=self.ip_port[0][0], home_joint=np.array([-100, -70, 125, -145, -93, 80]) / 180 * np.pi, gripper='blower')
            
            self.left_ur5 = UR5RTDE(self.ip_port[1][0], self.wsg50)
            self.right_ur5 = UR5RTDE(self.ip_port[2][0], 'rg2')
            self.ur5_pair = UR5PairRTDE(self.left_ur5, self.right_ur5)
            self.ur5_pair.open_gripper()
            if robot_home:
                self.ur5_pair.home()
                self.blow_ur5.home()
            self.blow_flag = False
            self.grasp_flag = False

            self.init_rot = Rotation.from_euler('z', -np.pi/2) * Rotation.from_rotvec([np.pi, 0, 0])
            self.left_rotvec = (Rotation.from_euler('z', -np.pi/2) * Rotation.from_rotvec([np.pi, 0, 0])).as_rotvec()
            self.right_rotvec = (Rotation.from_euler('z', 0) * Rotation.from_rotvec([np.pi, 0, 0])).as_rotvec()
            self.left_rotvec_tilt = (Rotation.from_euler('y', -np.pi/6) * Rotation.from_euler('z', -np.pi/2) * Rotation.from_rotvec([np.pi, 0, 0])).as_rotvec()
            self.right_rotvec_tilt = (Rotation.from_euler('y', np.pi/6) * Rotation.from_euler('z', 0) * Rotation.from_rotvec([np.pi, 0, 0])).as_rotvec()

            self.left_rotvec_pre_fling = (Rotation.from_euler('x', -np.pi/4) * Rotation.from_euler('z', -np.pi/2) * Rotation.from_rotvec([np.pi, 0, 0])).as_rotvec()
            self.right_rotvec_pre_fling = (Rotation.from_euler('x', -np.pi/4) * Rotation.from_euler('z', 0) * Rotation.from_rotvec([np.pi, 0, 0])).as_rotvec()

            self.left_rotvec_after_fling = (Rotation.from_euler('x', np.pi/4) * Rotation.from_euler('z', -np.pi/2) * Rotation.from_rotvec([np.pi, 0, 0])).as_rotvec()
            self.right_rotvec_after_fling = (Rotation.from_euler('x', np.pi/4) * Rotation.from_euler('z', 0) * Rotation.from_rotvec([np.pi, 0, 0])).as_rotvec()


            self.grasp_height = 0.034 + 0.028

            self.stretch_height = 0.4

            self.lift_height = 0.2
            # self.lift_y = -0.3
            self.lift_y = -0.16

            self.blow_height = 0.08
            # self.blow_y = -0.43
            self.blow_y = -0.29

            self.fling_height = 0.4

            self.primitive = primitive
            self.tilt = primitive == 'blow'

    def terminate(self):
        self.left_ur5.rtde_c.disconnect()
        self.right_ur5.rtde_c.disconnect()
        self.blow_ur5.robot.secmon.close()
        self.wsg50.bye()

    def get_current_cover_area(self, debug=False):
        self.color_img, self.depth_img = self.cam.get_camera_data(n=1)
        xyz_pts, color_pts = utils.get_pointcloud(self.depth_img, self.color_img, self.cam_intr, self.camera2table)
        if self.ee_pos is not None:
            invalid_idx = np.logical_and(
                np.logical_or(
                    xyz_pts[:, 0] < self.ee_pos[0][0] - 0.01,
                    xyz_pts[:, 0] > self.ee_pos[1][0] + 0.01
                ),
                xyz_pts[:, 2] > 0.18
            )
            color_pts[invalid_idx, :] = np.array([90, 90, 90])
        heightmap_bnd = np.array([[-0.512, 0.512], [-0.45, 0.574], [0, 1]])
        heightmap_pixel_size = 0.004
        heightmap, _ = utils.get_heightmap(xyz_pts, color_pts, heightmap_bnd, heightmap_pixel_size, 0)
        mask = utils.get_obj_mask(heightmap).astype(float)
        cover_area = np.sum(mask) * heightmap_pixel_size * heightmap_pixel_size
        self.cover_percentage = cover_area / self.max_cover_area
        if debug:
            return cover_area, heightmap
        else:
            return cover_area


    def get_observation(self, remove_robot=False):
        self.color_img, self.depth_img = self.cam.get_camera_data(n=1)

        # hacky: clean background
        obj_mask = utils.get_obj_mask(self.color_img)[:, :, np.newaxis]
        color_img = (np.ones_like(self.color_img) * 90 * (1 - obj_mask) + self.color_img * obj_mask).astype(np.uint8)
        depth_img = self.depth_img.copy()
        if remove_robot:
            xyz_pts = utils.get_pointcloud(self.depth_img, None, self.cam_intr, self.camera2table)[0].reshape(self.color_img.shape)
            kernel = np.array([
                [0, 1, 1, 0],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [0, 1, 1, 0]
            ], dtype=np.uint8)
            fg_mask = 1 - cv2.dilate((xyz_pts[:, :, 2] > self.lift_height+0.01).astype(np.float), kernel=kernel, iterations=5)[:, :, np.newaxis]

            pixel_coords = utils.project_pts_to_2d(np.asarray(self.ee_pos), self.table2camera, self.cam_intr).astype(int)
            fg_mask[pixel_coords[0][0] - 5: pixel_coords[1][0] + 5, 4400:] = 1
            color_img = (np.ones_like(color_img) * 90 * (1 - fg_mask) + color_img * fg_mask).astype(np.uint8)
            x_mean = (pixel_coords[0][1] + pixel_coords[1][1]) // 2 + 5
            color_img = cv2.circle(color_img, [self.color_img.shape[1] - 1 - x_mean, pixel_coords[0][0]], 2, (240, 240, 240), 5)
            color_img = cv2.circle(color_img, [self.color_img.shape[1] - 1 - x_mean, pixel_coords[1][0]], 2, (240, 240, 240), 5)

        color_img = np.rot90(color_img[self.crop_size[0][0]: self.crop_size[0][1], self.crop_size[1][0]: self.crop_size[1][1]])
        depth_img = np.rot90(depth_img[self.crop_size[0][0]: self.crop_size[0][1], self.crop_size[1][0]: self.crop_size[1][1]])
        observation = {
            'color_img': color_img,
            'depth_img': depth_img,
            'particle_view_color_img': np.rot90(self.color_img[:, 280:-280]),
            'particle_view_depth_img': np.rot90(self.depth_img[:, 280:-280])
        }
        return observation


    def reset_cloth(self):
        obs = self.get_observation()

        # start recording
        self.recording = True

        obj_mask = utils.get_obj_mask(obs['color_img'])
        non_zeros = np.stack(np.nonzero(obj_mask), 1)
        while True:
            pixel = non_zeros[np.random.choice(len(non_zeros))]

            dist = np.linalg.norm(pixel - np.array([obj_mask.shape[1] / 2, -obj_mask.shape[0] * 0.1]))
            if dist < obj_mask.shape[0] * 0.65:
                ur5 = self.left_ur5
                table2robot = self.table2left
                rotvec = self.left_rotvec
                break

        p = (self.translate_mat @ np.array([pixel[1], self.crop_info[1] - pixel[0], 1])).astype(int)[:2]
        wrd_p = utils.pixel_to_3d(self.depth_img, np.array([p]), self.camera2table, self.cam_intr)[0]
        ur5.movel(np.concatenate([utils.apply_transformation(np.array([wrd_p[0], wrd_p[1], 0.25]), table2robot), rotvec]))
        
        wrd_p[2] = self.grasp_height
        ur5.movel(np.concatenate([utils.apply_transformation(wrd_p, table2robot), rotvec]))
        ur5.close_gripper()
        wrd_p = np.array([0, -0.1, 0.5])
        ur5.movel(np.concatenate([utils.apply_transformation(wrd_p, table2robot), rotvec]))
        ur5.open_gripper()
        ur5.home()

        # stop recording
        self.recording = False


    def reset(self, *args):
        for _ in range(2):
            self.reset_cloth()

        # self.max_cover_area = 0.7*0.7 # Large Rect (red)
        # self.max_cover_area = 0.76*0.81 # X-Large Rect (pink)
        # self.max_cover_area = 0.395 # Shirt (blue)
        # self.max_cover_area = 0.43 # Dress (pink)

        self.max_cover_area = 0.395 # TODO: repalce this number
        self.grasp_cover_percentage_threshold = 0.8
        self.blow_cover_percentage_threshold = 0.7

        # TODO: just for demo, won't use it in paper
        if self.primitive == 'blow':
            self.grasp_cover_percentage_threshold += 0.05
            self.blow_cover_percentage_threshold += 0.05



        cover_area = self.get_current_cover_area()
        observation = self.get_observation()

        print('init cover percentage = ', self.cover_percentage)

        return self.max_cover_area, cover_area, observation

    def stretch_cloth(self):
        self.max_stretch_distance = 1.1 if self.primitive == 'blow' else 0.7
        while True:
            rgb, depth = self.realsense.get_camera_data()
            # if both arms no longer are holding onto cloth anymore
            grasp_flag = all(is_cloth_grasped(depth=depth))
            if not grasp_flag or is_cloth_stretched(rgb=rgb, depth=depth):
                return grasp_flag

            eps = 0.02
            self.ee_pos[0][0] -= eps
            self.ee_pos[1][0] += eps
            self.move_to(tilt=self.tilt)

            if self.ee_pos[1][0] - self.ee_pos[0][0] > self.max_stretch_distance:
                return all(is_cloth_grasped(depth=depth))

    def move_to_single(self, move_robot, target, middle=None):
        table2robot = self.table2left if move_robot == 'left' else self.table2right
        ee_rotvec = self.left_rotvec if move_robot == 'left' else self.right_rotvec
        robot = self.left_ur5 if move_robot == 'left' else self.right_ur5

        target = utils.apply_transformation(target, table2robot)
        target = np.concatenate([target, ee_rotvec]).tolist()

        if middle is not None:
            middle = utils.apply_transformation(middle, table2robot)
            middle = np.concatenate([middle, ee_rotvec]).tolist()
            robot.movel([middle, target])
        else:
            robot.movel(target)


    def move_to(self, wrd_p1=None, wrd_p2=None, tilt=False, middle_left=None, middle_right=None, pre_fling=False, after_fling=False, speed=1.5, acceleration=1):
        wrd_p1 = wrd_p1 if wrd_p1 is not None else self.ee_pos[0]
        wrd_p2 = wrd_p2 if wrd_p2 is not None else self.ee_pos[1]
        p_left = utils.apply_transformation(wrd_p1, self.table2left)
        p_right = utils.apply_transformation(wrd_p2, self.table2right)

        left_rotvec = self.left_rotvec_tilt if tilt else self.left_rotvec
        right_rotvec = self.right_rotvec_tilt if tilt else self.right_rotvec
        if pre_fling:
            left_rotvec = self.left_rotvec_pre_fling
            right_rotvec = self.right_rotvec_pre_fling

        p_left = np.concatenate([p_left, left_rotvec]).tolist()
        p_right = np.concatenate([p_right, right_rotvec]).tolist()

        if middle_left is not None:
            middle_left_rotvec = self.left_rotvec_after_fling if after_fling else left_rotvec
            middle_left = np.concatenate([utils.apply_transformation(middle_left, self.table2left), middle_left_rotvec]).tolist()
            p_left = [middle_left, p_left]
        if middle_right is not None:
            middle_right_rotvec = self.right_rotvec_after_fling if after_fling else right_rotvec
            middle_right = np.concatenate([utils.apply_transformation(middle_right, self.table2right), middle_right_rotvec]).tolist()
            p_right = [middle_right, p_right]

        self.ur5_pair.movel(p_left, p_right, speed, acceleration)

    
    def move_blower(self, position, orientation):
        position = utils.apply_transformation(np.array(position), self.table2blow)
        orientation = (Rotation.from_euler('z', orientation[0]) * Rotation.from_euler('x', orientation[2] + np.pi/2) * self.init_rot).as_rotvec()
        pose = np.concatenate([position, orientation])
        self.blow_ur5.movel(pose, speed=1.5, acceleration=0.5)


    def lift_and_stretch_primitive(self, p1, p2, placeholder=0):
        print('==> start lift', p1, p2)

        # start recording
        self.recording = True

        termination = False
        if p1 is None or p2 is None:
            print('terminate!!!')
            termination = True
            p1, p2 = [100, 100], [100, -100]

        # p1, p2: image_coordiante
        p1 = (self.translate_mat @ np.array([p1[1], self.crop_info[1] - p1[0], 1])).astype(int)[:2]
        p2 = (self.translate_mat @ np.array([p2[1], self.crop_info[1] - p2[0], 1])).astype(int)[:2]

        wrd_p1, wrd_p2 = utils.pixel_to_3d(self.depth_img, np.array([p1, p2]), self.camera2table, self.cam_intr)
        if wrd_p1[0] > wrd_p2[0]:
            wrd_p1, wrd_p2 = wrd_p2, wrd_p1
        wrd_p1[0] += 0.02 # TODO: hacky
        self.ee_pos = [wrd_p1.copy(), wrd_p2.copy()]

        pos_diff = np.abs(self.ee_pos[0] - self.ee_pos[1])
        if pos_diff[0] < 0.1 and pos_diff[1] < 0.25:
            termination = True
            print('Terminate: Too close!!!')
            
        if self.ee_pos[0][0] < -0.55 or self.ee_pos[1][0] > 0.55:
            termination = True
            print('Terminate: Grasping edge!!!')

        if self.cover_percentage > self.grasp_cover_percentage_threshold:
            termination = True
            print('Terminate: cover_percentage = ', self.cover_percentage)


        p_left = utils.apply_transformation(self.ee_pos[0], self.table2left)
        p_right = utils.apply_transformation(self.ee_pos[1], self.table2right)
        dist_left = np.linalg.norm(p_left)
        dist_right = np.linalg.norm(p_right)
        if min(dist_left, dist_right) < 0.15:
            termination = True
            print('Terminate: Close to base!')
        print(dist_left, dist_right)
        if max(dist_left, dist_right) > 1.0:
            termination = True
            print('Terminate: Too far to reach!')

        if not termination:
            self.ee_pos[0][2] = self.ee_pos[1][2] = 0.25

            middle_left = None if self.ee_pos[0][1] < -0.3 else np.array([-0.3, -0.4, 0.25])
            middle_right = None if self.ee_pos[1][1] < -0.3 else np.array([0.3, -0.4, 0.25])
            self.move_to(middle_left=middle_left, middle_right=middle_right)
            self.ee_pos[0][2] = self.grasp_height
            self.ee_pos[1][2] = np.clip(wrd_p2[2] - 0.015, self.grasp_height, self.grasp_height + 0.03)
            self.move_to()
            self.ur5_pair.close_gripper()

        lift_observation = self.get_observation()

        if not termination:
            distance = np.linalg.norm(self.ee_pos[0] - self.ee_pos[1])
            y_mid = (self.ee_pos[0][1] + self.ee_pos[1][1]) / 2
            self.ee_pos[0][0] = -distance / 2
            self.ee_pos[1][0] = distance / 2
            self.ee_pos[0][1] = self.ee_pos[1][1] = y_mid
            self.ee_pos[0][2] = self.ee_pos[1][2] = self.stretch_height
            self.move_to(tilt=self.tilt)
            self.ee_pos[0][1] = self.ee_pos[1][1] = self.lift_y
            self.move_to(tilt=self.tilt)

            self.grasp_flag = self.stretch_cloth()


        if self.grasp_flag and not termination:
            if self.primitive == 'blow':
                # turn on blower first
                position = np.array([0, self.blow_y, self.blow_height])
                orientation = np.array([0, 0, -95 / 180 * np.pi])
                self.move_blower(position, orientation)
                self.blow_ur5.open_gripper()
                time.sleep(0.2)

                self.ee_pos[0][2] = self.ee_pos[1][2] = self.lift_height
                self.move_to(tilt=self.tilt)
        else:
            self.ur5_pair.open_gripper()
            self.ur5_pair.home()

        # stop recording
        self.recording = False

        cover_area = self.get_current_cover_area()
        stretch_observation = self.get_observation(remove_robot=True)

        print('==> finish lift')

        return lift_observation, stretch_observation, cover_area

    
    def pick_and_place(self, p1, p2, placeholder=0):
        # start recording
        self.recording = True

        termination = False
        if p1 is None or p2 is None:
            print('terminate!!!')
            termination = True
            p1, p2 = [100, 100], [100, -100]

        # p1, p2: image_coordiante
        p1 = (self.translate_mat @ np.array([p1[1], self.crop_info[1] - p1[0], 1])).astype(int)[:2]
        p2 = (self.translate_mat @ np.array([p2[1], self.crop_info[1] - p2[0], 1])).astype(int)[:2]

        wrd_p1, wrd_p2 = utils.pixel_to_3d(self.depth_img, np.array([p1, p2]), self.camera2table, self.cam_intr)

        move_robot, middle_position = None, None

        p1_left = utils.apply_transformation(wrd_p1, self.table2left)
        p2_left = utils.apply_transformation(wrd_p2, self.table2left)
        dist_p1 = np.linalg.norm(p1_left)
        dist_p2 = np.linalg.norm(p2_left)
        if min(dist_p1, dist_p2) > 0.15 and max(dist_p1, dist_p2) < 0.85:
            move_robot = 'left'
            robot = self.left_ur5
            middle_position = None if wrd_p1[1] < -0.3 else np.array([-0.3, -0.4, 0.25])

        p1_right = utils.apply_transformation(wrd_p1, self.table2right)
        p2_right = utils.apply_transformation(wrd_p2, self.table2right)
        dist_p1 = np.linalg.norm(p1_right)
        dist_p2 = np.linalg.norm(p2_right)
        if min(dist_p1, dist_p2) > 0.15 and max(dist_p1, dist_p2) < 0.85:
            move_robot = 'right'
            robot = self.right_ur5
            middle_position = None if wrd_p1[1] < -0.3 else np.array([0.3, -0.4, 0.25])

        if move_robot is None:
            print('No Suitable Robot')
            termination = True

        if not termination:
            grasp_height = self.grasp_height if move_robot == 'left' else np.clip(wrd_p1[2] - 0.015, self.grasp_height, self.grasp_height + 0.03)
            wrd_p1[2] = 0.25
            self.move_to_single(move_robot, wrd_p1, middle_position)
            wrd_p1[2] = grasp_height
            if wrd_p1[1] > 0.2:
                wrd_p1[2] += 0.005
            self.move_to_single(move_robot, wrd_p1)
            robot.close_gripper()

        lift_observation = self.get_observation()

        if not termination:
            wrd_p1[2] = self.lift_height
            self.move_to_single(move_robot, wrd_p1)
            wrd_p2[2] = self.lift_height
            self.move_to_single(move_robot, wrd_p2)
            wrd_p2[2] = grasp_height
            self.move_to_single(move_robot, wrd_p2)
            robot.open_gripper()
            wrd_p2[2] = 0.25
            self.move_to_single(move_robot, wrd_p2)
            robot.home()
        
        # stop recording
        self.recording = False

        cover_area = self.get_current_cover_area()
        stretch_observation = self.get_observation()
        return lift_observation, stretch_observation, cover_area

    def place(self):
        # start recording
        self.recording = True

        print('place: ', self.grasp_flag, self.primitive)
        
        if self.grasp_flag and self.primitive != 'pick_and_place':
            if self.primitive == 'blow':
                time.sleep(0.2)
                self.blow_ur5.close_gripper()
                time.sleep(0.1)
                self.blow_ur5.home(blocking=False)

                self.ee_pos[0][1] = self.ee_pos[1][1] = self.lift_y - 0.1
                self.ee_pos[0][2] = self.ee_pos[1][2] = self.grasp_height + 0.01
                self.move_to(tilt=self.tilt)

            self.ur5_pair.open_gripper()
            self.ee_pos[0][2] = self.ee_pos[1][2] = 0.2
            self.move_to(tilt=False)
            self.ur5_pair.home()
            self.ee_pos = None
            self.grasp_flag = False

        
        # stop recording
        self.recording = False

        cover_area = self.get_current_cover_area()
        observation = self.get_observation()

        print('==> cover_percentage = ', cover_area / self.max_cover_area)

        return cover_area, observation

    def get_random_grasping(self, num_pair=100):
        obs = self.get_observation()
        color_img = obs['color_img']
        obj_mask = utils.get_obj_mask(color_img)
        non_zeros = np.stack(np.nonzero(obj_mask), axis=1)

        idx = np.random.choice(len(non_zeros), 2 * num_pair)
        select_positions = non_zeros[idx, :].reshape([2, num_pair, 2])

        distance = np.linalg.norm(select_positions[0] - select_positions[1], axis=1)
        pair_idx = np.argmax(distance)
        p1 = select_positions[0, pair_idx]
        p2 = select_positions[1, pair_idx]
        
        return p1, p2

    
    def get_random_pick_and_place(self, num_pair=100):
        obs = self.get_observation()
        color_img = obs['color_img']
        obj_mask = utils.get_obj_mask(color_img)
        non_zeros = np.stack(np.nonzero(obj_mask), axis=1)

        idx = np.random.choice(len(non_zeros))
        p1 = non_zeros[idx]
        direction = np.random.rand() * 2 * np.pi
        distance = np.random.rand() * 20 + 20
        p2 = p1 + distance * np.array([np.cos(direction), np.sin(direction)])
        return p1.astype(int), p2.astype(int)

    def blow(self, position, orientation, debug=False):
        # start recording
        self.recording = True

        if self.grasp_flag and self.cover_percentage < self.blow_cover_percentage_threshold:
            position = np.array([position[0], self.blow_y, self.blow_height])
            orientation[2] = -95 / 180 * np.pi
            self.move_blower(position, orientation)
            self.blow_ur5.open_gripper()
            time.sleep(0.2)

            if debug:
                self.blow_ur5.close_gripper()
        else:
            self.blow_ur5.close_gripper()
        
        # stop recording
        self.recording = False

        cover_area = self.get_current_cover_area()
        observation = None if debug else self.get_observation(remove_robot=True)
        print('==> finish blow. cover percentage = ', self.cover_percentage)
        return cover_area, observation


    def fling(self, debug=False):
        # start recording
        self.recording = True
        pos_diff = np.abs(self.ee_pos[0] - self.ee_pos[1])
        if self.grasp_flag and np.linalg.norm(pos_diff) > 0.15:
            self.ee_pos[0][1] = self.ee_pos[1][1] = self.lift_y
            self.ee_pos[0][2] = self.ee_pos[1][2] = self.fling_height
            self.move_to(pre_fling=True)

            middle_left = self.ee_pos[0].copy()
            middle_right = self.ee_pos[1].copy()
            middle_left[1] = middle_right[1] = self.lift_y + 0.6
            self.ee_pos[0][2] = self.ee_pos[1][2] = self.grasp_height + 0.02

            self.move_to(after_fling=True, middle_left=middle_left, middle_right=middle_right, speed=1.2, acceleration=5)

        # stop recording
        self.recording = False

        cover_area = None if debug else self.get_current_cover_area()
        observation = None if debug else self.get_observation(remove_robot=True)
        print('==> finish fling. cover percentage = ', self.cover_percentage)
        return cover_area, observation