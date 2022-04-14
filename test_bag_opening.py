import argparse
import os
import pickle
import time
import random

import numpy as np
import torch
from tqdm import trange

import wandb
from model import BagModel
from real_world.bag_env import BagEnv
from train_bag_opening import get_input_dim
from utils import mkdir


def main(args):
    wandb_name = f'{args.test_name}-{args.policy}'

    if args.test_num == 50:
        wandb.init(
            project='bag-opening-test',
            name=wandb_name
        )
        wandb.config.update(args)

    data_path = os.path.join('data', f'{args.test_name}.pkl')
    tasks = pickle.load(open(data_path, 'rb'))

    results_path = '.' # TODO: change the result path
    mkdir(results_path)
    
    print(f'==> Policy = {args.policy}')
    real_env = BagEnv(
        robot=True,
        realsense=True,
        robot_home=False,
        bag_type=args.bag_type
    )
    real_env.blow_ur5.close_gripper()

    if args.policy == 'learn':
        device = torch.device('cuda:0')
        input_dim, image_dim = get_input_dim(args.input_type)
        model = BagModel(input_dim=input_dim, image_dim=image_dim).to(device)
        if args.checkpoint.endswith('.pth'):
            checkpoint = torch.load(args.checkpoint)
        else:
            checkpoint = torch.load(os.path.join('exp', args.checkpoint, 'models', 'bag_latest.pth'))
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        torch.set_grad_enabled(False)

    results = [[] for _ in range(args.step_num + 1)]

    for idx in trange(args.test_num):
        log_path = os.path.join(results_path, f'test-{idx}')
        mkdir(log_path)
        grasp_position = tasks[idx]['grasp_position']
        distance = tasks[idx]['distance']
        tilt = tasks[idx]['tilt']
        blow_position = tasks[idx]['blow_position']
        reset_angle = tasks[idx]['reset_angle']

        real_env.move_gripper(grasp_position, distance, tilt)

        init_open_flag = False
        if args.policy != 'shake':
            real_env.move_blower([0, blow_position[0], blow_position[1]], [0, 0, reset_angle-15/180*np.pi])
            real_env.blow_ur5.open_gripper()
            time.sleep(1)
            real_env.move_blower([0, blow_position[0], blow_position[1]], [0, 0, reset_angle])
            init_open_flag, init_area_list, _ = real_env.get_reward()
            real_env.blow_ur5.close_gripper()
            time.sleep(1)

        log = dict()
        log['task'] = tasks[idx]
        log['init_open_flag'] = init_open_flag
        log['init_area_list'] = init_area_list
        results[0].append(init_open_flag)


        if args.policy == 'fixed':
            blow_position = np.mean(real_env.blow_space, axis=1)[:2]
            center_grasping_position = np.mean(real_env.grasping_space, axis=1)[:2]
            angle = np.arctan2(center_grasping_position[1] - blow_position[1], center_grasping_position[0] - blow_position[0])
            real_env.move_blower(
                position=[0, blow_position[0], blow_position[1]],
                orientation=[0, 0, angle],
                speed=0.05, acceleration=0.1
            )

            real_env.blow_ur5.open_gripper()
            open_flag, area_list, raw_image_list = real_env.get_reward()
            
            log[f'area_list'] = area_list
            log[f'open_flag'] = open_flag

            for step_id in range(args.step_num):
                results[step_id+1].append(open_flag)

        elif args.policy == 'learn':
            # initial pose            
            blow_position = real_env.get_random_blow_position()
            angle = np.arctan2(grasp_position[1] - 0.2 - blow_position[1], grasp_position[0] - blow_position[0])
            real_env.move_blower([0, blow_position[0], blow_position[1]], [0, 0, angle])
            real_env.blow_ur5.open_gripper()
            time.sleep(2)

            open_flag = False
            for step_id in range(args.step_num):
                if open_flag:
                    results[step_id + 1].append(open_flag)
                    continue
                observation = real_env.get_observation()

                depth_image = torch.from_numpy(observation[1][100:512+100, 500:512+500].astype(np.float32)[np.newaxis, np.newaxis, ...]).to(device)
                color_image = torch.from_numpy(observation[0][100:512+100, 500:512+500].astype(np.float32).transpose([2, 0, 1])[np.newaxis, ...]  / 255.0).to(device)
                if 'rgb' in args.input_type:
                    if 'depth' in args.input_type:
                        observation = torch.cat([color_image, depth_image], dim=1)
                    else:
                        observation = color_image
                else:
                    observation = depth_image

                action_input_list = list()
                blow_action_candidates = list()
                current_action = real_env.get_current_action()

                for _ in range(args.action_num):
                    inputs = list()
                    blow_position = real_env.get_random_blow_position()
                    angle = random.uniform(real_env.blow_space[2][0], real_env.blow_space[2][1])
                    blow_action = np.array([blow_position[0], blow_position[1], angle])
                    blow_action_candidates.append(blow_action)
                    
                    if 'abs' in args.input_type:
                        inputs.extend([blow_action[0], blow_action[1], blow_action[2]])
                    if 'curr' in args.input_type:
                        inputs.extend([current_action[0], current_action[1], current_action[2]])
                    action_input_list.append(torch.from_numpy(np.array(inputs, dtype=np.float32)[np.newaxis]).to(device))
                pred = model(observation, action_input_list)
                pred = [x.item() for x in pred]
                max_idx = np.argmax(pred)
                blow_action = blow_action_candidates[max_idx]
                angle = np.arctan2(grasp_position[1] - blow_action[1], grasp_position[0] - blow_action[0]) / np.pi * 180

                log[f'observation-{step_id}'] = observation
                log[f'current_action-{step_id}'] = current_action
                log[f'blow_action_candidates-{step_id}'] = blow_action_candidates
                log[f'pred-{step_id}'] = pred
                log[f'max_idx-{step_id}'] = max_idx
                log[f'blow_action-{step_id}'] = blow_action
                
                real_env.move_blower(
                    position=[0, blow_action[0], blow_action[1]],
                    orientation=[0, 0, blow_action[2]],
                    speed=0.05, acceleration=0.1
                )
                real_env.blow_ur5.open_gripper()
                open_flag, area_list, raw_image_list = real_env.get_reward()

                log[f'area_list-{step_id}'] = area_list
                log[f'open_flag-{step_id}'] = open_flag
                results[step_id+1].append(open_flag)

        elif args.policy == 'shake':
            real_env.labeling = True
            real_env.shake(tilt)
            real_env.labeling = False
            raw_image_list, area_list = list(), list()
            for color_img, depth_img in real_env.label_images:
                raw_image_list.append(color_img)
                area = real_env.get_area(color_img, depth_img)
                area_list.append(area)
            real_env.label_images = list()
            max_area = np.max(area_list)
            open_flag = max_area > real_env.area_threshold

            log[f'area_list'] = area_list
            log[f'open_flag'] = open_flag
            for step_id in range(args.step_num):
                results[step_id+1].append(open_flag)
        else:
            raise NotImplementedError()
            
        real_env.blow_ur5.close_gripper()
        
        time.sleep(1)
        pickle.dump(log, open(os.path.join(log_path, 'log.pkl'), 'wb'))

        real_env.all_label_images = [list(), list()]

        idx += 1

        print([np.mean(x) for x in results])

    data = list()
    for i in range(args.step_num + 1):
        data.append([i, np.mean(results[i])])

    if args.test_num == 50:
        table = wandb.Table(data=data, columns = ["step", "acc"])
        wandb.log({"acc" : wandb.plot.line(table, "step", "acc", title="acc")})

    real_env.blow_ur5.close_gripper()
    real_env.terminate()

if __name__=='__main__':
    parser = argparse.ArgumentParser('Test bag opening')
    # exp & dataset
    parser.add_argument('--test_name', type=str, default='bag-rss', help='test_name')
    parser.add_argument('--test_num', type=int, default=50, help='test num')
    parser.add_argument('--step_num', type=int, default=4, help='step num')
    parser.add_argument('--bag_type', type=str, default='white', help='bag type')
    parser.add_argument('--action_num', type=int, default=64, help='action num')
    parser.add_argument('--policy', type=str, default='learn', choices=['learn', 'fixed', 'shake'], help='policy')
    parser.add_argument('--checkpoint', type=str, default='bag-opening', help='data name')
    args = parser.parse_args()

    args.input_type = ['depth', 'abs', 'curr']

    main(args)