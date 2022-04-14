import argparse
import os
import shutil
import time

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import ray
import torch
from filelock import FileLock
from tqdm import trange

import utils
from utils import get_obj_mask, get_line_mask, str2bool
import wandb
from sim_env import SimEnv
from model import GraspModel, BlowModel


class GraspDataset(torch.utils.data.Dataset):
    def __init__(self, replay_buffer_path):
        self.replay_buffer_path = replay_buffer_path
        with h5py.File(self.replay_buffer_path, 'r') as data:
            self.data_len = int(np.array(data['curr_data_size']))

    def __len__(self):
        return self.data_len
    
    def __getitem__(self, idx):
        with h5py.File(self.replay_buffer_path, 'r') as data:
            group = data[f'data-{idx}']
            color_image = np.array(group['color_image']).astype(np.float32)
            observation = color_image.transpose([2, 0, 1])
            cover_area = np.array(group['cover_area']).astype(np.float32)
            init_cover_area = np.array(group['init_cover_area']).astype(np.float32)
            reward = cover_area - init_cover_area
            grasp_center = np.array(group['grasp_center']).astype(int)
            grasp_angle = np.array(group['grasp_angle']).astype(np.float32)
        return observation, grasp_center, grasp_angle, reward


class BlowDataset(torch.utils.data.Dataset):
    def __init__(self, replay_buffer_path):
        self.replay_buffer_path = replay_buffer_path
        with h5py.File(self.replay_buffer_path, 'r') as data:
            self.data_len = int(np.array(data['curr_data_size']))

    def __len__(self):
        return self.data_len
    
    def __getitem__(self, idx):
        with h5py.File(self.replay_buffer_path, 'r') as data:
            group = data[f'data-{idx}']
            color_image = np.array(group['color_image']).astype(np.float32)
            observation = color_image.transpose([2, 0, 1])
            cover_area = np.array(group['cover_area']).astype(np.float32)
            reward = cover_area
            action = np.array(group['blow_action']).astype(np.float32)
            last_action = np.array(group['last_blow_action']).astype(np.float32)
        return observation, action, last_action, reward


def generate_line_masks(rotation_num, resolution):
    rotation_angles = np.arange(rotation_num)*(np.pi / rotation_num)
    masks = list()
    for angle in rotation_angles:
        img = np.zeros([resolution * 2 - 1, resolution * 2 - 1])
        center = np.array([resolution - 1, resolution - 1])
        direction = np.array([np.cos(angle), np.sin(angle)])
        p0 = center + direction * resolution * 2
        p1 = center - direction * resolution * 2
        masks.append({
            'angle': angle,
            'direction': direction,
            'mask': cv2.line(img, [int(p0[1]), int(p0[0])], [int(p1[1]), int(p1[0])], 1, 1)
        })
    return masks

def add_data(group, name, data, compression=False):
    if name in group:
        group[name][...] = data
    else:
        if compression:
            group.create_dataset(name=name, data=data, compression='gzip', compression_opts=5)
        else:
            group.create_dataset(name=name, data=data)


def get_blow_actions(action_candidates, score_candidates, epsilon):
    actions, scores = list(), list()
    for i in range(score_candidates.shape[1]):
        idx = np.argmax(score_candidates[:, i]) if np.random.rand() > epsilon else np.random.choice(score_candidates.shape[0])
        actions.append(action_candidates[idx, i])
        scores.append(score_candidates[idx, i])
    return actions, scores

def get_grasp_action(affordance_maps, line_masks, obj_mask, args):
    # affordance_maps: [N, W, H]
    valid_mask = np.ones([args.grasp_resolution, args.grasp_resolution])
    sorted_actions = np.stack(np.unravel_index(np.argsort(-affordance_maps, axis=None), affordance_maps.shape), axis=1)
    for action in sorted_actions:
        pixel = action[1:]
        if valid_mask[pixel[0], pixel[1]] == 0:
            continue
        angle_id = action[0]
        direction = line_masks[angle_id]['direction']
        line_mask = get_line_mask(line_masks, pixel, angle_id, args.grasp_resolution)
        mask = line_mask * obj_mask
        if np.max(mask) == 0:
            continue
        pixel_candidates = np.stack(np.nonzero(mask), axis=1)
        dist = np.sum((pixel_candidates - np.array(pixel)) * direction, axis=1)
        p0 = pixel_candidates[np.argmin(dist)]
        p1 = pixel_candidates[np.argmax(dist)]
        if np.max(dist) < 0 or np.min(dist) > 0 or np.linalg.norm(p0 - p1) < 4:
            continue
        if p0[1] > p1[1]:
            p0, p1 = p1, p0
        d0 = np.linalg.norm(p0 - np.array([(args.grasp_resolution - 1) / 2, -args.grasp_resolution * 0.1]))
        d1 = np.linalg.norm(p1 - np.array([(args.grasp_resolution - 1) / 2, args.grasp_resolution * (1+0.1)]))
        if max(d0, d1) < args.grasp_resolution * 0.6:
            return action
        valid_mask *= (1 - line_mask)
    return None

def get_flingbot_action(affordance_maps, obj_mask, args):
    sorted_actions = np.stack(np.unravel_index(np.argsort(-affordance_maps, axis=None), affordance_maps.shape), axis=1)
    for action in sorted_actions:
        scale_id = action[0]
        angle_id = action[1]
        pixel = action[2:]

        angle = args.grasp_angle_options[angle_id]
        scale = args.grasp_scale_options[scale_id]
        p0 = (pixel + np.array([np.cos(angle), np.sin(angle)]) * 8).astype(int)
        p1 = (pixel - np.array([np.cos(angle), np.sin(angle)]) * 8).astype(int)

        mat = utils.get_transform_matrix(obj_mask.shape[0], args.grasp_resolution, 1.0/scale)
        pix0 = (np.array([p0[0], p0[1], 1]) @ mat).astype(int)[:2]
        pix1 = (np.array([p1[0], p1[1], 1]) @ mat).astype(int)[:2]
        if np.min(pix0) < 0 or np.max(pix0) >= obj_mask.shape[0] or np.min(pix1) < 0 or np.max(pix1) >= obj_mask.shape[0]:
            continue
        if obj_mask[pix0[0], pix0[1]] + obj_mask[pix1[0], pix1[1]] == 1:
            continue

        mat = utils.get_transform_matrix(args.grasp_resolution, args.grasp_resolution, 1.0/scale)
        p0 = (np.array([p0[0], p0[1], 1]) @ mat).astype(int)[:2]
        p1 = (np.array([p1[0], p1[1], 1]) @ mat).astype(int)[:2]

        if p0[1] > p1[1]:
            p0, p1 = p1, p0
        d0 = np.linalg.norm(p0 - np.array([(args.grasp_resolution - 1) / 2, -args.grasp_resolution * 0.1]))
        d1 = np.linalg.norm(p1 - np.array([(args.grasp_resolution - 1) / 2, args.grasp_resolution * (1+0.1)]))
        if max(d0, d1) < args.grasp_resolution * 0.6:
            return action
    return action


def get_pick_and_place_action(affordance_maps, obj_mask, args):
    sorted_actions = np.stack(np.unravel_index(np.argsort(-affordance_maps, axis=None), affordance_maps.shape), axis=1)
    for action in sorted_actions:
        scale_id = action[0]
        angle_id = action[1]
        pixel = action[2:]

        angle = args.grasp_angle_options[angle_id]
        scale = args.grasp_scale_options[scale_id]
        p0 = pixel.astype(int)
        p1 = (pixel + np.array([np.cos(angle), np.sin(angle)]) * 16).astype(int)

        mat = utils.get_transform_matrix(obj_mask.shape[0], args.grasp_resolution, 1.0/scale)
        pix0 = (np.array([p0[0], p0[1], 1]) @ mat).astype(int)[:2]
        pix1 = (np.array([p1[0], p1[1], 1]) @ mat).astype(int)[:2]
        if np.min(pix0) < 0 or np.max(pix0) >= obj_mask.shape[0] or obj_mask[pix0[0], pix0[1]] == 0:
            continue
        if np.min(pix1) < 0 or np.max(pix1) >= obj_mask.shape[0] or obj_mask[pix1[0], pix1[1]] == 1:
            continue

        mat = utils.get_transform_matrix(args.grasp_resolution, args.grasp_resolution, 1.0/scale)
        p0 = (np.array([p0[0], p0[1], 1]) @ mat).astype(int)[:2]
        p1 = (np.array([p1[0], p1[1], 1]) @ mat).astype(int)[:2]

        d0 = np.linalg.norm(p0 - np.array([(args.grasp_resolution - 1) / 2, -args.grasp_resolution * 0.1]))
        d1 = np.linalg.norm(p1 - np.array([(args.grasp_resolution - 1) / 2, -args.grasp_resolution * 0.1]))
        if max(d0, d1) < args.grasp_resolution * 0.6:
            return action

        d0 = np.linalg.norm(p0 - np.array([(args.grasp_resolution - 1) / 2, args.grasp_resolution * (1+0.1)]))
        d1 = np.linalg.norm(p1 - np.array([(args.grasp_resolution - 1) / 2, args.grasp_resolution * (1+0.1)]))
        if max(d0, d1) < args.grasp_resolution * 0.6:
            return action
    return action


def collect_data(envs, args, line_masks, task_ids,
        grasp_model, grasp_device, grasp_replay_buffer_path, grasp_epsilon,
        blow_model, blow_device, blow_replay_buffer_path, blow_epsilon, real_env=False):
    # torch preparation
    if grasp_model is not None:
        grasp_model.eval()
    if blow_model is not None:
        blow_model.eval()
    torch.set_grad_enabled(False)

    # reset
    max_cover_area, cover_area, init_observation = utils.reset_envs(envs, args.task, args.task_num, task_ids)
    data_sequence = list()

    for grasp_step in trange(args.grasp_step_num):
        grasp_init_cover_area = cover_area
        data = {
            'init_cover_area': grasp_init_cover_area,
            'init_cover_percentage': [x / y for x, y in zip(cover_area, max_cover_area)]
        }
        data_sequence.append(data)

        # grasping
        grasping_info = list()
        grasping_actions = list()
        if args.grasp_policy == 'random':
            grasping_actions = utils.get_grasping_acitons(envs)
        elif args.grasp_policy == 'heuristic':
            grasp_image_input = [cv2.resize(obs['color_img'], (args.grasp_resolution, args.grasp_resolution)) for obs in init_observation]
            scene_input = np.stack(grasp_image_input).transpose([0, 3, 1, 2]).astype(np.float32)
            scene_input = torch.from_numpy(scene_input).to(grasp_device)
            affordance_maps = grasp_model(scene_input).cpu().numpy() # [B, N, W, H]
            data['affordance_maps'] = affordance_maps
            for i in range(len(envs)):
                depth_image = init_observation[i]['depth_img']
                obj_mask = get_obj_mask(grasp_image_input[i])
                if np.random.rand() > grasp_epsilon:
                    action = get_grasp_action(affordance_maps[i], line_masks, obj_mask, args)
                    if action is None:
                        angle_id = 0
                        pixel = np.array([0, 0])
                    else:
                        angle_id = action[0]
                        pixel = action[1:]
                else:
                    non_zeros = np.stack(np.nonzero(obj_mask), axis=1)
                    pixel_id = np.random.choice(len(non_zeros))
                    pixel = non_zeros[pixel_id]
                    angle_id = np.random.choice(args.grasp_rotation_num)
                score = affordance_maps[i, angle_id, pixel[0], pixel[1]]
                direction = line_masks[angle_id]['direction']
                obj_mask = get_obj_mask(grasp_image_input[i])
                line_mask = get_line_mask(line_masks, pixel, angle_id, args.grasp_resolution)
                mask = line_mask * obj_mask

                valid_action = True
                if np.max(mask) == 0:
                    valid_action = False
                else:
                    pixel_candidates = np.stack(np.nonzero(mask), axis=1)
                    dist = np.sum((pixel_candidates - np.array(pixel)) * direction, axis=1)
                    p0 = pixel_candidates[np.argmin(dist)]
                    p1 = pixel_candidates[np.argmax(dist)]
                    if np.max(dist) < 0 or np.min(dist) > 0 or np.linalg.norm(p0 - p1) < 4:
                        valid_action = False       

                if valid_action:
                    mat = utils.get_transform_matrix(depth_image.shape[0], grasp_image_input[i].shape[0], 1)
                    pix0 = (mat @ np.array([p0[0], p0[1], 1])).astype(int)[:2]
                    pix1 = (mat @ np.array([p1[0], p1[1], 1])).astype(int)[:2]

                    if real_env:
                        grasping_actions.append([pix0, pix1])
                    else:
                        wrd_p0, wrd_p1 = utils.pixel_to_3d(depth_image, np.array([pix0, pix1]), args.cam_pose, args.cam_intr)
                        if wrd_p0[0] < wrd_p1[0]:
                            wrd_p0, wrd_p1 = wrd_p1, wrd_p0
                        grasping_actions.append([wrd_p0, wrd_p1])
                else:
                    p0 = (pixel + args.grasp_resolution // 10 * direction).astype(int)
                    p1 = (pixel - args.grasp_resolution // 10 * direction).astype(int)
                    grasping_actions.append([[2, 1, 0], [2, 1, 0]])
                if real_env:
                    img = grasp_image_input[0]
                    img = cv2.circle(img, [p0[1], p0[0]], 2, (0,0,0), 2)
                    img = cv2.circle(img, [p1[1], p1[0]], 2, (0,0,0), 2)
                    utils.imwrite('color_img.png', img)
                    # input('enter!')
                    
                grasping_info.append({
                    'scale': 1.0,
                    'angle_id': angle_id,
                    'angle': line_masks[angle_id]['angle'],
                    'center': pixel,
                    'score': score,
                    'end_points': [p0, p1],
                    'succ': valid_action
                })
        elif args.grasp_policy == 'flingbot':
            affordance_maps = list()
            for scale in args.grasp_scale_options:
                crop_dim = int(init_observation[0]['color_img'].shape[0] / scale)
                scale_imgs = [utils.crop_center(obs['color_img'], crop_dim) for obs in init_observation]
                image_input = [cv2.resize(img, (args.grasp_resolution, args.grasp_resolution)) for img in scale_imgs]
                scene_input = np.stack(image_input).transpose([0, 3, 1, 2]).astype(np.float32)
                scene_input = torch.from_numpy(scene_input).to(grasp_device)
                affordance_maps.append(grasp_model(scene_input).cpu().numpy()) # [B, S, W, H]
            affordance_maps = np.stack(affordance_maps, axis=1) # [B, S, R, W, H]
            data['affordance_maps'] = affordance_maps
            grasp_image_input = list()
            for i in range(len(envs)):
                color_image = init_observation[i]['color_img']
                depth_image = init_observation[i]['depth_img']
                obj_mask = get_obj_mask(color_image)
                if np.random.rand() > grasp_epsilon:
                    # action = np.array(np.unravel_index(np.argmax(affordance_maps[i]), affordance_maps[i].shape))
                    action = get_flingbot_action(affordance_maps[i], obj_mask, args)
                    scale_id = action[0]
                    angle_id = action[1]
                    pixel = action[2:]
                else:
                    scale_id = np.random.choice(len(args.grasp_scale_options))
                    angle_id = np.random.choice(args.grasp_rotation_num)
                    pixel = np.random.choice(args.grasp_resolution, 2)
                score = affordance_maps[i, scale_id, angle_id, pixel[0], pixel[1]]
                angle = args.grasp_angle_options[angle_id]
                scale = args.grasp_scale_options[scale_id]
                p0 = (pixel + np.array([np.cos(angle), np.sin(angle)]) * 8).astype(int)
                p1 = (pixel - np.array([np.cos(angle), np.sin(angle)]) * 8).astype(int)
                crop_dim = int(color_image.shape[0] / scale)
                scale_img = utils.crop_center(color_image, crop_dim)
                grasp_image_input.append(cv2.resize(scale_img, (args.grasp_resolution, args.grasp_resolution)))

                mat = utils.get_transform_matrix(depth_image.shape[0], args.grasp_resolution, 1.0/scale)
                pix0 = (np.array([p0[0], p0[1], 1]) @ mat).astype(int)[:2]
                pix1 = (np.array([p1[0], p1[1], 1]) @ mat).astype(int)[:2]
                valid_action = np.min([pix0[0], pix0[1], pix1[0], pix1[1]]) >= 0 and np.max([pix0[0], pix0[1], pix1[0], pix1[1]]) < depth_image.shape[0]
                
                # if score < 0.005:
                #     valid_action = False

                if valid_action:
                    if real_env:
                        grasping_actions.append([pix0, pix1])
                    else:
                        wrd_p0, wrd_p1 = utils.pixel_to_3d(depth_image, np.array([pix0, pix1]), args.cam_pose, args.cam_intr)
                        if wrd_p0[0] < wrd_p1[0]:
                            wrd_p0, wrd_p1 = wrd_p1, wrd_p0
                        grasping_actions.append([wrd_p0, wrd_p1])
                else:
                    grasping_actions.append([[2, 1, 0], [2, 1, 0]])

                if real_env:
                    img = grasp_image_input[0]
                    img = cv2.circle(img, [p0[1], p0[0]], 2, (0,0,0), 2)
                    img = cv2.circle(img, [p1[1], p1[0]], 2, (0,0,0), 2)
                    utils.imwrite('color_img.png', img)
                    # input('enter!')

                grasping_info.append({
                    'scale': scale,
                    'angle_id': angle_id,
                    'angle': line_masks[angle_id]['angle'],
                    'center': pixel,
                    'score': score,
                    'end_points': [p0, p1],
                    'succ': valid_action
                })
        elif args.grasp_policy == 'pick_and_place':
            affordance_maps = list()
            for scale in args.grasp_scale_options:
                crop_dim = int(init_observation[0]['color_img'].shape[0] / scale)
                scale_imgs = [utils.crop_center(obs['color_img'], crop_dim) for obs in init_observation]
                image_input = [cv2.resize(img, (args.grasp_resolution, args.grasp_resolution)) for img in scale_imgs]
                scene_input = np.stack(image_input).transpose([0, 3, 1, 2]).astype(np.float32)
                scene_input = torch.from_numpy(scene_input).to(grasp_device)
                affordance_maps.append(grasp_model(scene_input).cpu().numpy()) # [B, S, W, H]
            affordance_maps = np.stack(affordance_maps, axis=1) # [B, S, R, W, H]
            data['affordance_maps'] = affordance_maps
            grasp_image_input = list()
            for i in range(len(envs)):
                color_image = init_observation[i]['color_img']
                depth_image = init_observation[i]['depth_img']
                obj_mask = get_obj_mask(color_image)
                if np.random.rand() > grasp_epsilon:
                    # action = np.array(np.unravel_index(np.argmax(affordance_maps[i]), affordance_maps[i].shape))
                    action = get_pick_and_place_action(affordance_maps[i], obj_mask, args)
                    scale_id = action[0]
                    angle_id = action[1]
                    pixel = action[2:]
                else:
                    scale_id = np.random.choice(len(args.grasp_scale_options))
                    angle_id = np.random.choice(args.grasp_rotation_num)
                    pixel = np.random.choice(args.grasp_resolution, 2)
                score = affordance_maps[i, scale_id, angle_id, pixel[0], pixel[1]]
                angle = args.grasp_angle_options[angle_id]
                scale = args.grasp_scale_options[scale_id]
                p0 = pixel.astype(int)
                p1 = (pixel + np.array([np.cos(angle), np.sin(angle)]) * 16).astype(int)
                crop_dim = int(color_image.shape[0] / scale)
                scale_img = utils.crop_center(color_image, crop_dim)
                grasp_image_input.append(cv2.resize(scale_img, (args.grasp_resolution, args.grasp_resolution)))

                mat = utils.get_transform_matrix(depth_image.shape[0], args.grasp_resolution, 1.0/scale)
                pix0 = (np.array([p0[0], p0[1], 1]) @ mat).astype(int)[:2]
                pix1 = (np.array([p1[0], p1[1], 1]) @ mat).astype(int)[:2]
                valid_action = np.min([pix0[0], pix0[1], pix1[0], pix1[1]]) >= 0 and np.max([pix0[0], pix0[1], pix1[0], pix1[1]]) < depth_image.shape[0]
                # print(i, obj_mask[pix0[0], pix0[1]], obj_mask[pix1[0], pix1[1]], color_image[pix0[0], pix0[1]], color_image[pix1[0], pix1[1]])
                # if score < 0.005:
                #     valid_action = False

                if valid_action:
                    if real_env:
                        grasping_actions.append([pix0, pix1])
                    else:
                        wrd_p0, wrd_p1 = utils.pixel_to_3d(depth_image, np.array([pix0, pix1]), args.cam_pose, args.cam_intr)
                        grasping_actions.append([wrd_p0, wrd_p1])
                else:
                    grasping_actions.append([[2, 1, 0], [2, 1, 0]])

                if real_env:
                    img = grasp_image_input[0]
                    img = cv2.circle(img, [p0[1], p0[0]], 2, (0,0,0), 2)
                    img = cv2.circle(img, [p1[1], p1[0]], 2, (0,0,0), 2)
                    utils.imwrite('color_img.png', img)
                    # input('enter!')

                grasping_info.append({
                    'scale': scale,
                    'angle_id': angle_id,
                    'angle': line_masks[angle_id]['angle'],
                    'center': pixel,
                    'score': score,
                    'end_points': [p0, p1],
                    'succ': valid_action
                })
        else:
            raise NotImplementedError(f'Grasp policy does not support \"{args.grasp_policy}\"')

        if args.grasp_policy == 'pick_and_place':
            lift_observation, stretch_observation, cover_area = utils.pick_and_place(envs, grasping_actions, lifting_height=0.15)
        else:
            lift_observation, stretch_observation, cover_area = utils.lift_and_stretch(envs, grasping_actions, lifting_height=0.12)

        data['grasping_info'] = grasping_info
        data['init_observation'] = init_observation
        data['lift_observation'] = lift_observation
        data['stretch_observation'] = stretch_observation

        if args.blow_policy == 'fling':
            cover_area, observation = utils.fling(envs)
            data[f'blow_observation'] = observation
            data[f'blow_cover_area'] = cover_area
            data[f'blow_cover_percentage'] = [x / y for x, y in zip(cover_area, max_cover_area)]
        elif args.blow_policy == 'box':
            cover_area, observation = utils.blow_box(envs, 120)
            data[f'blow_observation'] = observation
            data[f'blow_cover_area'] = cover_area
            data[f'blow_cover_percentage'] = [x / y for x, y in zip(cover_area, max_cover_area)]
        elif args.blow_policy == 'fixed':
            current_observation = stretch_observation
            # rx_list = [-30, 0, 30]
            rx_list = [0]
            for blow_step, rx in enumerate(rx_list):
                blow_init_cover_area = cover_area
                blow_actions = [np.array([0, 0.03, 0.45, rx / 180 * np.pi, 0, -105 / 180 * np.pi]) for env in envs]
                image_input = [cv2.resize(obs['color_img'], (args.grasp_resolution, args.grasp_resolution)) for obs in current_observation]
                
                cover_area, blow_observation = utils.blow(envs, blow_actions, args.blow_time)
                data[f'blow_observation-{blow_step}'] = blow_observation
                data[f'blow_observation_input-{blow_step}'] = image_input
                data[f'blow_cover_area-{blow_step}'] = cover_area
                data[f'blow_cover_percentage-{blow_step}'] = [x / y for x, y in zip(cover_area, max_cover_area)]
                data[f'blow_init_cover_area-{blow_step}'] = blow_init_cover_area
                data[f'blow_init_cover_percentage-{blow_step}'] = [x / y for x, y in zip(blow_init_cover_area, max_cover_area)]
                data[f'blow_action-{blow_step}'] = blow_actions

                current_observation = blow_observation

        elif args.blow_policy == 'learn':
            current_observation = stretch_observation
            last_blow_actions = np.zeros([len(envs), 6])
            for blow_step in range(args.blow_step_num):
                blow_init_cover_area = cover_area
                image_input = [cv2.resize(obs['color_img'], (args.blow_resolution, args.blow_resolution)) for obs in current_observation]

                if args.blow_last_action and blow_step == 0:
                    action_candidates, score_candidates = None, None
                    blow_actions, scores = blow_model.get_forward_actions(len(envs))
                else:
                    scene_input = np.stack(image_input).transpose([0, 3, 1, 2]).astype(np.float32)
                    scene_input = torch.from_numpy(scene_input).to(blow_device)
                    last_action = torch.from_numpy(np.array(last_blow_actions).astype(np.float32)).to(blow_device)
                    action_candidates, score_candidates = blow_model(scene_input, None, last_action)
                    score_candidates = score_candidates.cpu().numpy()
                    blow_actions, scores = get_blow_actions(action_candidates, score_candidates, blow_epsilon)

                cover_area, blow_observation = utils.blow(envs, blow_actions, args.blow_time)
                data[f'blow_observation-{blow_step}'] = blow_observation
                data[f'blow_observation_input-{blow_step}'] = image_input
                data[f'blow_actions-{blow_step}'] = action_candidates
                data[f'blow_scores-{blow_step}'] = score_candidates
                data[f'blow_cover_area-{blow_step}'] = cover_area
                data[f'blow_cover_percentage-{blow_step}'] = [x / y for x, y in zip(cover_area, max_cover_area)]
                data[f'blow_init_cover_area-{blow_step}'] = blow_init_cover_area
                data[f'blow_init_cover_percentage-{blow_step}'] = [x / y for x, y in zip(blow_init_cover_area, max_cover_area)]
                data[f'blow_action-{blow_step}'] = blow_actions
                data[f'blow_score-{blow_step}'] = scores

                if blow_replay_buffer_path is not None and (blow_step != 0 or not args.blow_last_action):
                    with FileLock(blow_replay_buffer_path + '.lock'):
                        with h5py.File(blow_replay_buffer_path, 'a') as dataset:
                            for i in range(len(envs)):
                                max_data_size = int(np.array(dataset['max_data_size']))
                                curr_data_size = min(int(np.array(dataset['curr_data_size'])) + 1, max_data_size)
                                current_idx = (int(np.array(dataset['current_idx'])) + 1) % max_data_size
                                dataset['current_idx'][...] = current_idx
                                dataset['curr_data_size'][...] = curr_data_size
                                group = dataset[f'data-{current_idx}'] if f'data-{current_idx}' in dataset.keys() else dataset.create_group(f'data-{current_idx}')
                                add_data(group, 'blow_action', blow_actions[i], False)
                                add_data(group, 'last_blow_action', last_blow_actions[i], False)
                                add_data(group, 'cover_area', cover_area[i], False)
                                add_data(group, 'init_cover_area', blow_init_cover_area[i], False)
                                add_data(group, 'color_image', image_input[i], True)
                last_blow_actions = blow_actions
                current_observation = blow_observation
        elif args.blow_policy is None:
            pass
        else:
            raise NotImplementedError(f'Blow policy does not support \"{args.blow_policy}\"')

        cover_area, final_observation = utils.place(envs)
        data[f'final_observation'] = final_observation
        data[f'cover_area'] = cover_area
        data[f'cover_percentage'] = [x / y for x, y in zip(cover_area, max_cover_area)]

        if grasp_replay_buffer_path is not None and args.grasp_policy in ['flingbot', 'heuristic', 'pick_and_place']:
            with FileLock(grasp_replay_buffer_path + '.lock'):
                with h5py.File(grasp_replay_buffer_path, 'a') as dataset:
                    for i in range(len(envs)):
                        max_data_size = int(np.array(dataset['max_data_size']))
                        curr_data_size = min(int(np.array(dataset['curr_data_size'])) + 1, max_data_size)
                        current_idx = (int(np.array(dataset['current_idx'])) + 1) % max_data_size
                        dataset['current_idx'][...] = current_idx
                        dataset['curr_data_size'][...] = curr_data_size
                        group = dataset[f'data-{current_idx}'] if f'data-{current_idx}' in dataset.keys() else dataset.create_group(f'data-{current_idx}')
                        add_data(group, 'grasp_center', grasping_info[i]['center'], False)
                        add_data(group, 'grasp_angle', grasping_info[i]['angle'], False)
                        add_data(group, 'init_cover_area', grasp_init_cover_area[i], False)
                        add_data(group, 'cover_area', cover_area[i], False)
                        add_data(group, 'color_image', grasp_image_input[i], True)
        init_observation = final_observation
    return data_sequence


def visualization(args, data_sequence, line_masks, vis_path, title):
    cmap = plt.get_cmap('jet')
    html_data = {}
    ids = [f'{i}-{j}' for i in range(args.visualization_num) for j in range(args.grasp_step_num)]
    cols = ['init', 'grasp', 'lift', 'stretch', 'final']
    if args.blow_policy in ['learn', 'fixed']:
        for blow_step in range(args.blow_step_num):
            cols.append(f'blow_score-{blow_step}')
            cols.append(f'blow_obs-{blow_step}')
            cols.append(f'blow_particle-{blow_step}')
    if args.grasp_policy == 'heuristic':
        cols += [f'affordance-{angle_id}' for angle_id in range(len(args.grasp_angle_options))]
    elif args.grasp_policy in ['flingbot', 'pick_and_place']:
        cols += [f'affordance-{angle_id}-{scale_id}' for angle_id in range(len(args.grasp_angle_options)) for scale_id in range(len(args.grasp_scale_options))]

    for grasp_step in range(args.grasp_step_num):
        data = data_sequence[grasp_step]
        for env_id in range(args.visualization_num):
            depth_image = data['init_observation'][env_id]['depth_img']
            color_image = data['init_observation'][env_id]['color_img']

            html_data[f'{env_id}-{grasp_step}_init'] = color_image

            grasp_img = color_image.copy()
            text_scale = 1 if grasp_img.shape[0] == 720 else 2/3
            text_p1 = (np.array([25, 50]) * text_scale).astype(int)
            text_p2 = (np.array([25, 100]) * text_scale).astype(int)
            text_p3 = (np.array([25, 150]) * text_scale).astype(int)
            id_p = [grasp_img.shape[0] - int(70 * text_scale), int(100 * text_scale)]
            fontScale = 1.5 * text_scale
            thickness = int(3 * text_scale)

            if args.grasp_policy in ['heuristic', 'flingbot', 'pick_and_place']:
                pixel = data['grasping_info'][env_id]['center']
                scale = data['grasping_info'][env_id]['scale']
                angle_id = data['grasping_info'][env_id]['angle_id']
                p0, p1 = data['grasping_info'][env_id]['end_points']
                color = (0, 0, 0) if data['grasping_info'][env_id]['succ'] else (255, 255, 255)

                mat = utils.get_transform_matrix(depth_image.shape[0], args.grasp_resolution, 1.0/scale)
                pixel = (np.array([pixel[0], pixel[1], 1]) @ mat).astype(int)[:2]
                p0 = (np.array([p0[0], p0[1], 1]) @ mat).astype(int)[:2]
                p1 = (np.array([p1[0], p1[1], 1]) @ mat).astype(int)[:2]
                
                grasp_img = cv2.circle(grasp_img, [pixel[1], pixel[0]], 9, color, 9)
                grasp_img = cv2.circle(grasp_img, [p0[1], p0[0]], 6, color, 6)
                grasp_img = cv2.circle(grasp_img, [p1[1], p1[0]], 6, color, 6)
                grasp_img = cv2.line(grasp_img, [pixel[1], pixel[0]], [p0[1], p0[0]], color, 6)
                grasp_img = cv2.line(grasp_img, [pixel[1], pixel[0]], [p1[1], p1[0]], color, 6)
                grasp_img = cv2.putText(grasp_img, f'[{pixel[0]}, {pixel[1]}] / {angle_id}', text_p1, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=fontScale, color=(255,255,255), thickness=thickness)
            
            html_data[f'{env_id}-{grasp_step}_grasp'] = grasp_img
            html_data[f'{env_id}-{grasp_step}_lift'] = data['lift_observation'][env_id]['color_img']
            html_data[f'{env_id}-{grasp_step}_stretch'] = data['stretch_observation'][env_id]['color_img']

            final_img = data['final_observation'][env_id]['color_img'].copy()
            score = data['grasping_info'][env_id]['score'] if args.grasp_policy in ['heuristic', 'flingbot', 'pick_and_place'] else -1
            cover_area = data['cover_area'][env_id]
            delta_area = data['cover_area'][env_id] - data['init_cover_area'][env_id]
            cover_percentage = data['cover_percentage'][env_id]
            delta_percentage = data['cover_percentage'][env_id] - data['init_cover_percentage'][env_id]
            final_img = cv2.putText(final_img, f'score:{score:.3f}', text_p1, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=fontScale, color=(255,255,255), thickness=thickness)
            final_img = cv2.putText(final_img, f'area:{cover_area:.3f} ({delta_area:.3f})', text_p2, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=fontScale, color=(255,255,255), thickness=thickness)
            final_img = cv2.putText(final_img, f'ratio:{cover_percentage:.3f} ({delta_percentage:.3f})', text_p3, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=fontScale, color=(255,255,255), thickness=thickness)
            html_data[f'{env_id}-{grasp_step}_final'] = final_img

            if args.grasp_policy == 'heuristic':
                obj_mask = get_obj_mask(cv2.resize(color_image, (args.grasp_resolution, args.grasp_resolution)))
                affordance_maps = data['affordance_maps'][env_id]
                affordance_maps_normalized = affordance_maps.copy()
                affordance_maps_normalized /= np.max(np.abs(affordance_maps_normalized))
                affordance_maps_normalized = affordance_maps_normalized / 2 + 0.5

                for angle_id in range(len(args.grasp_angle_options)):
                    pixel = np.array(np.unravel_index(np.argmax(affordance_maps[angle_id]), affordance_maps[angle_id].shape))
                    vis_affordance_map = cmap(affordance_maps_normalized[angle_id])[:, :, :3] * 0.8 + obj_mask[:, :, np.newaxis] * 0.2

                    line_mask = get_line_mask(line_masks, pixel, angle_id, args.grasp_resolution)
                    direction = line_masks[angle_id]['direction']
                    mask = line_mask * obj_mask

                    valid_action = True
                    if np.max(mask) == 0:
                        valid_action = False
                    else:
                        pixel_candidates = np.stack(np.nonzero(mask), axis=1)
                        dist = np.sum((pixel_candidates - np.array(pixel)) * direction, axis=1)
                        p0 = pixel_candidates[np.argmin(dist)]
                        p1 = pixel_candidates[np.argmax(dist)]
                        if np.max(dist) < 0 or np.min(dist) > 0 or np.linalg.norm(p0 - p1) < 4:
                            valid_action = False

                    vis_affordance_map = (vis_affordance_map * 255).astype(np.uint8)
                    color = (0, 0, 0) if valid_action else (255, 255, 255)
                    vis_affordance_map = cv2.circle(vis_affordance_map, [pixel[1], pixel[0]], 3, color, 3)
                    if valid_action:
                        vis_affordance_map = cv2.circle(vis_affordance_map, [p0[1], p0[0]], 2, color, 2)
                        vis_affordance_map = cv2.circle(vis_affordance_map, [p1[1], p1[0]], 2, color, 2)
                        vis_affordance_map = cv2.line(vis_affordance_map, [pixel[1], pixel[0]], [p0[1], p0[0]], color, 2)
                        vis_affordance_map = cv2.line(vis_affordance_map, [pixel[1], pixel[0]], [p1[1], p1[0]], color, 2)

                    vis_affordance_map = cv2.putText(vis_affordance_map, f'score:{np.max(affordance_maps[angle_id]):.3f}', (8, 18), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(255,255,255), thickness=1)
                    html_data[f'{env_id}-{grasp_step}_affordance-{angle_id}'] = vis_affordance_map
            elif args.grasp_policy in ['flingbot', 'pick_and_place']:
                affordance_maps = data['affordance_maps'][env_id]
                affordance_maps_normalized = affordance_maps.copy()
                affordance_maps_normalized /= np.max(np.abs(affordance_maps_normalized))
                affordance_maps_normalized = affordance_maps_normalized / 2 + 0.5
                for scale_id in range(len(args.grasp_scale_options)):
                    scale = args.grasp_scale_options[scale_id]
                    crop_dim = int(color_image.shape[0] / scale)
                    obj_mask = get_obj_mask(cv2.resize(utils.crop_center(color_image, crop_dim), (args.grasp_resolution, args.grasp_resolution)))
                    for angle_id in range(len(args.grasp_angle_options)):
                        vis_affordance_map = cmap(affordance_maps_normalized[scale_id, angle_id])[:, :, :3] * 0.8 + obj_mask[:, :, np.newaxis] * 0.2
                        html_data[f'{env_id}-{grasp_step}_affordance-{angle_id}-{scale_id}'] = vis_affordance_map
            
            if args.blow_policy in ['fixed', 'learn']:
                for blow_step in range(args.blow_step_num):
                    action = data[f'blow_action-{blow_step}'][env_id]
                    cover_area = data[f'blow_cover_area-{blow_step}'][env_id]
                    cover_percentage = data[f'blow_cover_percentage-{blow_step}'][env_id]
                    delta_area = data[f'blow_cover_area-{blow_step}'][env_id] - data[f'blow_init_cover_area-{blow_step}'][env_id]
                    delta_percentage = data[f'blow_cover_percentage-{blow_step}'][env_id] - data[f'blow_init_cover_percentage-{blow_step}'][env_id]

                    blow_score_bg = np.tile(get_obj_mask(data[f'blow_observation_input-{blow_step}'][env_id])[:, :, np.newaxis], 3)
                    blow_score_img = np.zeros_like(blow_score_bg)
                    if args.blow_policy == 'learn' and data[f'blow_actions-{blow_step}'] is not None:
                        cmap = plt.get_cmap('jet')
                        actions = data[f'blow_actions-{blow_step}'][:, env_id]
                        scores = data[f'blow_scores-{blow_step}'][:, env_id]
                        scores -= np.min(scores)
                        scores /= max(np.max(scores), 0.1)
                        action_color = cmap(scores)[:, :3]
                        for k in range(args.blow_action_sample_num):
                            angle = actions[k][3]+np.pi
                            st = np.array([202, 127.5 + actions[k][0] * 120]).astype(int)
                            fi = (st + np.array([np.cos(angle), np.sin(angle)]) * 150).astype(int)
                            blow_score_img = cv2.line(blow_score_img, [st[1], st[0]], [fi[1], fi[0]], action_color[k], 1)
                    blow_score_img = ((blow_score_img * 0.8 + blow_score_bg * 0.2) * 255).astype(np.uint8)
                    html_data[f'{env_id}-{grasp_step}_blow_score-{blow_step}'] = blow_score_img

                    obs_img = data[f'blow_observation-{blow_step}'][env_id]['color_img'].copy()
                    if args.blow_policy == 'learn':
                        score = data[f'blow_score-{blow_step}'][env_id]
                        obs_img = cv2.putText(obs_img, f'score:{score:.3f}', text_p1, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=fontScale, color=(255,255,255), thickness=thickness)
                    obs_img = cv2.putText(obs_img, f'area:{cover_area:.3f} ({delta_area:.3f})', text_p2, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=fontScale, color=(255,255,255), thickness=thickness)
                    obs_img = cv2.putText(obs_img, f'ratio:{cover_percentage:.3f} ({delta_percentage:.3f})', text_p3, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=fontScale, color=(255,255,255), thickness=thickness)
                    obs_img = cv2.putText(obs_img, f'{blow_step}', id_p, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=fontScale*2, color=(255,255,255), thickness=thickness*2)
                    html_data[f'{env_id}-{grasp_step}_blow_obs-{blow_step}'] = obs_img

                    particle_img = data[f'blow_observation-{blow_step}'][env_id]['particle_view_color_img'].copy()
                    particle_img = cv2.putText(particle_img, f'p:{action[0]:.2f}, {action[1]:.2f}, {action[2]:.2f}', text_p1, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.5, color=(255,255,255), thickness=3)
                    particle_img = cv2.putText(particle_img, f'r:{action[3]/np.pi*180:.1f}, {action[4]/np.pi*180:.1f}, {action[5]/np.pi*180:.1f}', text_p2, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.5, color=(255,255,255), thickness=3)
                    particle_img = cv2.putText(particle_img, f'{blow_step}', (650, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(255,255,255), thickness=8)
                    html_data[f'{env_id}-{grasp_step}_blow_particle-{blow_step}'] = particle_img
    utils.html_visualize(vis_path, html_data, ids, cols, title=title, clean=False)


def main(args):
    # Set wandb
    wandb.init(
        project='cloth-unfolding-train',
        name=args.exp
    )
    wandb.config.update(args)

    # Save arguments
    exp_dir = os.path.join('exp', args.exp)
    utils.mkdir(exp_dir, clean=True)
    str_list = []
    for key in vars(args):
        print('[{0}] = {1}'.format(key, getattr(args, key)))
        str_list.append('--{0}={1} \\'.format(key, getattr(args, key)))
    with open(os.path.join('exp', args.exp, 'args.txt'), 'w+') as f:
        f.write('\n'.join(str_list))

    # Set directory. e.g. visualization, model snapshot
    args.visualization_dir = os.path.join('exp', args.exp, 'visualization')
    utils.mkdir(args.visualization_dir)
    args.model_dir = os.path.join('exp', args.exp, 'models')
    utils.mkdir(args.model_dir)

    # Set replay buffer
    grasp_replay_buffer_path = os.path.join('exp', args.exp, 'grasp_replay_buffer.hdf5')
    with h5py.File(grasp_replay_buffer_path, 'a') as data:
        data['max_data_size'] = args.grasp_replay_buffer_size; data['curr_data_size'] = 0; data['current_idx'] = -1

    blow_replay_buffer_path = os.path.join('exp', args.exp, 'blow_replay_buffer.hdf5')
    with h5py.File(blow_replay_buffer_path, 'a') as data:
        data['max_data_size'] = args.blow_replay_buffer_size; data['curr_data_size'] = 0; data['current_idx'] = -1

    # ray env
    grasp_device = torch.device(f'cuda:{args.grasp_gpu}' if torch.cuda.is_available() else 'cpu')
    blow_device = torch.device(f'cuda:{args.blow_gpu}' if torch.cuda.is_available() else 'cpu')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.pyflex_gpus
    num_pyflex_gpus = len(args.pyflex_gpus.split(','))
    ray.init()
    RaySimEnv = ray.remote(SimEnv).options(num_cpus=1, num_gpus=num_pyflex_gpus/args.env_num)
    envs = [RaySimEnv.remote(gui=False, wind_life_time=args.wind_life_time, large_grasp=False, pick_and_place=args.grasp_policy == 'pick_and_place') for _ in range(args.env_num)]

    # get camera matrix (intr, pose)
    args.cam_intr, args.cam_pose = ray.get(envs[0].get_camera_matrix.remote())

    # Set model & optimizer & criteria
    print('==> Preparing model & optimizer')
    if args.grasp_policy in ['heuristic', 'flingbot', 'pick_and_place']:
        grasp_model = GraspModel(model_type=args.grasp_policy, rotation_num=args.grasp_rotation_num).to(grasp_device)
        grasp_optimizer = torch.optim.Adam(grasp_model.parameters(), lr=args.grasp_learning_rate, weight_decay=args.grasp_weight_decay)
    else:
        grasp_model=None; grasp_optimizer = None
    
    if args.blow_policy == 'learn':
        blow_model = BlowModel(action_sample_num=args.blow_action_sample_num, x_range=args.blow_x_range, z_rotation=args.blow_z_rotation, last_action=args.blow_last_action).to(blow_device)
        blow_optimizer = torch.optim.Adam(blow_model.parameters(), lr=args.blow_learning_rate, weight_decay=args.blow_weight_decay)
    else:
        blow_model = None; blow_optimizer = None
    criteria = torch.nn.MSELoss()

    # Load checkpoint
    if args.grasp_checkpoint is not None:
        print(f'==> Loading grasping checkpoint from {args.grasp_checkpoint}')
        if args.grasp_checkpoint.endswith('.pth'):
            checkpoint = torch.load(args.grasp_checkpoint)
        else:
            checkpoint = torch.load(os.path.join('exp', args.grasp_checkpoint, 'models', 'grasp_latest.pth'))
        grasp_model.load_state_dict(checkpoint['state_dict'])
        grasp_optimizer.load_state_dict(checkpoint['optimizer'])
        print(f'==> Loaded grasping checkpoint from {args.grasp_checkpoint}')
    if args.blow_checkpoint is not None:
        print(f'==> Loading blowing checkpoint from {args.blow_checkpoint}')
        if args.blow_checkpoint.endswith('.pth'):
            checkpoint = torch.load(args.blow_checkpoint, map_location=grasp_device)
        else:
            checkpoint = torch.load(os.path.join('exp', args.blow_checkpoint, 'models', 'blow_latest.pth'), map_location=grasp_device)
        blow_model.load_state_dict(checkpoint['state_dict'])
        blow_optimizer.load_state_dict(checkpoint['optimizer'])
        print(f'==> Loaded blowing checkpoint from {args.blow_checkpoint}')

    line_masks = generate_line_masks(args.grasp_rotation_num, args.grasp_resolution)

    for epoch in range(args.epoch):
        grasp_epsilon = 0.1 if epoch > 90 or args.grasp_checkpoint is not None else 1.0 - epoch / 100
        blow_epsilon = 0.1 if epoch > 45 or args.blow_checkpoint is not None else 1.0 - epoch / 50

        wandb_info = {
            'exp-info/grasp-epsilon': grasp_epsilon,
            'exp-info/blow-epsilon': blow_epsilon
        }

        print(f'==> epoch = {epoch}, grasp_epsilon = {grasp_epsilon:.3f}, blow_epsilon = {blow_epsilon:.3f}')
        epoch_start_time = time.time()

        # collect data
        data_sequence = collect_data(
            envs, args, line_masks, None,
            grasp_model, grasp_device,  grasp_replay_buffer_path, grasp_epsilon,
            blow_model, blow_device,  blow_replay_buffer_path, blow_epsilon
        )
        collect_data_time = time.time() - epoch_start_time

        wandb_info[f'grasp-cover-percentage/init'] = np.nanmean(data_sequence[0]['init_cover_percentage'])
        if args.blow_policy in ['fixed', 'learn']:
            wandb_info[f'blow-cover-percentage/init'] = list()
            for blow_step in range(args.blow_step_num):
                wandb_info[f'blow-cover-percentage/step-{blow_step}'] = list()
        
        for grasp_step in range(args.grasp_step_num):
            wandb_info[f'grasp-cover-percentage/step-{grasp_step}'] = np.nanmean(data_sequence[grasp_step]['cover_percentage'])
            wandb_info[f'grasp-succ/step-{grasp_step}'] = np.nanmean([info['succ'] for info in data_sequence[grasp_step]['grasping_info']])

            if args.blow_policy in ['fixed', 'learn']:
                wandb_info[f'blow-cover-percentage/init'].append(np.nanmean(data_sequence[grasp_step][f'blow_init_cover_percentage-0']))
                for blow_step in range(args.blow_step_num):
                    wandb_info[f'blow-cover-percentage/step-{blow_step}'].append(np.nanmean(data_sequence[grasp_step][f'blow_cover_percentage-{blow_step}']))
        
        if args.blow_policy in ['fixed', 'learn']:
            wandb_info[f'blow-cover-percentage/init'] = np.nanmean(wandb_info[f'blow-cover-percentage/init'])
            for blow_step in range(args.blow_step_num):
                wandb_info[f'blow-cover-percentage/step-{blow_step}'] = np.nanmean(wandb_info[f'blow-cover-percentage/step-{blow_step}'])

        # train
        torch.set_grad_enabled(True)
        if grasp_model is not None:
            grasp_model.train()
            dataset = GraspDataset(grasp_replay_buffer_path)
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.grasp_batch_size, shuffle=True, num_workers=args.num_workers)
            data_loader_iter = iter(data_loader)
            train_loss, train_pred, train_gt, data_num = 0, 0, 0, 0
            for i in trange(min(len(data_loader), args.grasp_iter_per_epoch)):
                observation, grasp_center, grasp_angle, reward = next(data_loader_iter)
                batch_size = observation.size(0)
                if batch_size == 1:
                    continue
                grasp_center = grasp_center.numpy()
                grasp_angle = grasp_angle.numpy()
                pred = grasp_model(observation.to(grasp_device), [grasp_angle]) # [B, 1, W, H]

                pred = pred[np.arange(batch_size), 0, grasp_center[:, 0], grasp_center[:, 1]]
                gt = reward.to(grasp_device)
                loss = criteria(pred, gt) * 1000

                train_loss += loss.item() * batch_size
                train_pred += torch.sum(pred).item()
                train_gt += torch.sum(gt).item()
                data_num += batch_size

                grasp_optimizer.zero_grad()
                loss.backward()
                grasp_optimizer.step()
            
            train_loss /= data_num
            train_pred /= data_num
            train_gt /= data_num

            print(f'[Grasp] train loss = {train_loss:.4f}, pred = {train_pred:.4f}, gt = {train_gt:.4f}, replay buffer size = {len(dataset)}')
            wandb_info['training/grasp-loss'] = train_loss
            wandb_info['training/grasp-pred'] = train_pred
            wandb_info['training/grasp-gt'] = train_gt
            wandb_info['exp-info/grasp-replay-buffer-size'] = len(dataset)

        if blow_model is not None and epoch >= args.blow_freeze_epoch:
            dataset = BlowDataset(blow_replay_buffer_path)
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.blow_batch_size, shuffle=True, num_workers=args.num_workers)
            data_loader_iter = iter(data_loader)

            train_loss, train_pred, train_gt, data_num = 0, 0, 0, 0
            for i in trange(min(len(data_loader), args.blow_iter_per_epoch)):
                observation, action, last_action, reward = next(data_loader_iter)
                batch_size = observation.size(0)
                if batch_size == 1:
                    continue
                pred = blow_model(observation.to(blow_device), action.to(blow_device), last_action.to(blow_device))
                gt = reward.to(blow_device)
                loss = criteria(pred, gt) * 1000.0

                train_loss += loss.item() * batch_size
                train_pred += torch.sum(pred).item()
                train_gt += torch.sum(gt).item()
                data_num += batch_size

                blow_optimizer.zero_grad()
                loss.backward()
                blow_optimizer.step()
            
            train_loss /= data_num
            train_pred /= data_num
            train_gt /= data_num

            print(f'[Blow] train loss = {train_loss:.4f}, pred = {train_pred:.4f}, gt = {train_gt:.4f}, replay buffer size = {len(dataset)}')
            wandb_info['training/blow-loss'] = train_loss
            wandb_info['training/blow-pred'] = train_pred
            wandb_info['training/blow-gt'] = train_gt
            wandb_info['exp-info/blow-replay-buffer-size'] = len(dataset)
        wandb.log(wandb_info)


        total_time = time.time() - epoch_start_time
        train_time = total_time - collect_data_time
        print(f'{total_time:.2f}(total) = {collect_data_time:.2f}(data) + {train_time:.2f}(train)')

        if epoch == 0 or (epoch + 1) % args.snapshot_gap == 0:
            # visualization
            print('...visualizating...')
            data_sequence = collect_data(
                envs, args, line_masks, None,
                grasp_model, grasp_device,  grasp_replay_buffer_path, 0,
                blow_model, blow_device,  blow_replay_buffer_path, 0
            )
            vis_path = os.path.join(args.visualization_dir, 'epoch_%06d' % (epoch + 1))
            title = f'{epoch+1}-{args.exp}'
            visualization(args, data_sequence, line_masks, vis_path, title)

            # save checkpoint
            if grasp_model is not None:
                save_state = {
                    'state_dict': grasp_model.state_dict(),
                    'optimizer': grasp_optimizer.state_dict(),
                    'epoch': epoch + 1
                }
                torch.save(save_state, os.path.join(args.model_dir, 'grasp_latest.pth'))
                shutil.copyfile(
                    os.path.join(args.model_dir, 'grasp_latest.pth'),
                    os.path.join(args.model_dir, 'grasp_epoch_%06d.pth' % (epoch + 1))
                )
                
            if blow_model is not None:
                save_state = {
                    'state_dict': blow_model.state_dict(),
                    'optimizer': blow_optimizer.state_dict(),
                    'epoch': epoch + 1
                }
                torch.save(save_state, os.path.join(args.model_dir, 'blow_latest.pth'))
                shutil.copyfile(
                    os.path.join(args.model_dir, 'blow_latest.pth'),
                    os.path.join(args.model_dir, 'blow_epoch_%06d.pth' % (epoch + 1))
                )


if __name__=='__main__':
    parser = argparse.ArgumentParser('Grasp')
    # exp & dataset
    parser.add_argument('--exp', type=str, default=None, help='exp name')
    parser.add_argument('--task', default='Train_Normal_Rect', type=str, help='init state dataset path')
    parser.add_argument('--task_num', default=2000, type=int, help='number of init state')
    parser.add_argument('--epoch', default=1000, type=int, help='number of epoch')
    parser.add_argument('--num_workers', default=8, type=int, help='num_workers of data loader')
    parser.add_argument('--snapshot_gap', default=20, type=int, help='Frequence of saving the snapshot (e.g. visualization, model, optimizer)')
    parser.add_argument('--visualization_num', default=8, type=int, help='visualization num')

    # sim env
    parser.add_argument('--pyflex_gpus', type=str, default='0,1,2,3,4,5,6,7', help='pyflex gpu ids')
    parser.add_argument('--env_num', default=32, type=int, help='number of environment')
    parser.add_argument('--wind_life_time', default=60, type=int, help='wind life time')

    # policy
    parser.add_argument('--policy', default='DextAIRity', type=str, choices=['DextAIRity', 'DextAIRity_random_grasp', 'DextAIRity_fixed', 'FlingBot', 'FlingBot_plus', 'Pick_and_Place'], help='type of policy')

    # grasping
    parser.add_argument('--grasp_step_num', default=5, type=int, help='number of grasping steps')
    parser.add_argument('--grasp_rotation_num', default=8, type=int, help='number of arotations')

    parser.add_argument('--grasp_replay_buffer_size', type=int, default=30000, help='replay buffer size of grasping training')
    parser.add_argument('--grasp_gpu', type=str, default='0', help='grasping policy gpu id')
    parser.add_argument('--grasp_learning_rate', default=1e-4, type=float, help='learning rate of the grasp optimizer')
    parser.add_argument('--grasp_weight_decay', default=1e-6, type=float, help='weight decay of the grasp optimizer')
    parser.add_argument('--grasp_iter_per_epoch', default=64, type=int, help='grasp training iteration per epoch')
    parser.add_argument('--grasp_batch_size', default=16, type=int, help='grasp_batch size')
    parser.add_argument('--grasp_checkpoint', type=str, default=None, help='exp name of grasp policy checkpoint')
    

    # blowing
    parser.add_argument('--blow_step_num', default=4, type=int, help='number of grasping steps')
    parser.add_argument('--blow_time', default=150, type=int, help='number of steps of each blowing')
    parser.add_argument('--blow_freeze_epoch', default=0, type=int, help='number of epoch to freeze the blowing model')
    parser.add_argument('--blow_action_sample_num', default=64, type=int, help='number of action samples')
    parser.add_argument('--blow_x_range', default=0.1, type=float, help='x range')
    parser.add_argument('--blow_z_rotation', default=-95, type=float, help='z rotation')
    parser.add_argument('--blow_last_action', type=str2bool, nargs='?', const=True, default=False, help="Input last action")
    

    parser.add_argument('--blow_replay_buffer_size', type=int, default=30000, help='replay buffer size of blowing training')
    parser.add_argument('--blow_gpu', type=str, default='1', help='blowing policy gpu id')
    parser.add_argument('--blow_learning_rate', default=1e-4, type=float, help='learning rate of the blow optimizer')
    parser.add_argument('--blow_weight_decay', default=1e-6, type=float, help='weight decay of the blow optimizer')
    parser.add_argument('--blow_iter_per_epoch', default=64, type=int, help='blow training iteration per epoch')
    parser.add_argument('--blow_batch_size', default=128, type=int, help='blow batch size')
    parser.add_argument('--blow_checkpoint', type=str, default=None, help='exp name of blow policy checkpoint')

    args = parser.parse_args()

    # parse policy
    if args.policy == 'DextAIRity':
        args.grasp_policy = 'heuristic'
        args.blow_policy = 'learn'
    if args.policy == 'DextAIRity_random_grasp':
        args.grasp_policy = 'random'
        args.blow_policy = 'learn'
    elif args.policy == 'DextAIRity_fixed':
        args.grasp_policy = 'heuristic'
        args.blow_policy = 'fixed'
        args.blow_step_num = 1
    elif args.policy == 'FlingBot_plus':
        args.grasp_policy = 'heuristic'
        args.blow_policy = 'fling'
        args.blow_step_num = 1
    elif args.policy == 'FlingBot':
        args.grasp_policy = 'flingbot'
        args.blow_policy = 'fling'
        args.blow_step_num = 1
    elif args.policy == 'Pick_and_Place':
        args.grasp_policy = 'flingbot'
        args.blow_policy = None
        args.blow_step_num = 0

    # exp
    if args.exp is None:
        args.exp = args.policy
        
    # task dir
    args.task = os.path.join('data', args.task)

    # resolution of input image
    args.grasp_resolution = 256 if args.grasp_policy == 'heuristic' else 64
    args.blow_resolution = 256

    # flingbot parameters
    args.grasp_scale_options = [1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75]
    args.grasp_angle_options = np.arange(args.grasp_rotation_num) * (2 * np.pi / args.grasp_rotation_num) if args.grasp_policy == 'pick_and_place' else np.arange(args.grasp_rotation_num) * (np.pi / args.grasp_rotation_num)

    main(args)