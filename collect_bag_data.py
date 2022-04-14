import argparse
import os
import time
import pickle

import numpy as np
from tqdm import tqdm
import random

import utils
from real_world.bag_env import BagEnv


def main(args):
    data_root = os.path.join('data', args.data_name)
    utils.mkdir(data_root, clean=False)

    real_env = BagEnv(
        robot=True,
        realsense=True,
        robot_home=False,
        bag_type=args.bag_type
    )

    pbar = tqdm(total=args.data_num)
    pbar.update(args.start_id)
    idx = args.start_id
    positive_num = 0
    while idx < args.data_num:
        real_env.blow_ur5.close_gripper()
        grasp_position, distance, tilt = real_env.get_random_grasp_position()
        real_env.move_gripper(grasp_position, distance, tilt)

        blow_position = real_env.get_random_blow_position()
        angle = np.arctan2(grasp_position[1] - 0.2 - blow_position[1], grasp_position[0] - blow_position[0])
        real_env.move_blower([0, blow_position[0], blow_position[1]], [0, 0, angle])
        real_env.blow_ur5.open_gripper()
        time.sleep(2)
        
        for _ in range(5):
            observation = real_env.get_observation()
            angle = random.uniform(real_env.blow_space[2][0], real_env.blow_space[2][1])
            real_env.move_blower([0, blow_position[0], blow_position[1]], [0, 0, angle], speed=0.05, acceleration=0.1)
            open_flag, area_list, raw_image_list = real_env.get_reward()

            data = {
                'observation': observation,
                'grasp_info': (grasp_position, distance, tilt),
                'current_action': real_env.get_current_action(),
                'open_flag': open_flag,
                'area_list': area_list,
                'bag_type': args.bag_type
            }
            pickle.dump(data, open(os.path.join(data_root, f'data-{idx}.pkl'), 'wb'))
            positive_num += open_flag
            idx += 1

            pbar.update()
            pbar.set_description(f'p_num={positive_num}; p_rate={positive_num/(idx - args.start_id):.2f}')
            if open_flag or idx >= args.data_num:
                break

    real_env.terminate()

if __name__=='__main__':
    parser = argparse.ArgumentParser('Bag data collection')
    # exp & dataset
    parser.add_argument('--data_name', type=str, default='bag_opening', help='data name')
    parser.add_argument('--data_num', type=int, default=4400, help='data num')
    parser.add_argument('--start_id', type=int, default=0, help='start id')
    parser.add_argument('--bag_type', type=str, default='white', help='bag type')

    args = parser.parse_args()

    main(args)
