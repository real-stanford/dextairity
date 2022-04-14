import argparse
import os
import pickle

import numpy as np
import torch

from model import BlowModel, GraspModel
from real_world.cloth_env import RealWorldEnv
from train_cloth_unfolding import (collect_data, generate_line_masks,
                                   visualization)
from utils import mkdir, str2bool
import matplotlib.pyplot as plt


def main(args):
    grasp_device = torch.device(f'cuda:{args.grasp_gpu}' if torch.cuda.is_available() else 'cpu')
    blow_device = torch.device(f'cuda:{args.blow_gpu}' if torch.cuda.is_available() else 'cpu')
    
    envs = [RealWorldEnv(primitive='blow' if args.blow_policy == 'learn' else args.blow_policy)]

    args.visualization_dir = os.path.join('exp', args.exp, 'test-real' + args.suffix)

    mkdir(args.visualization_dir, clean=True)
    
    # Set model
    print('==> Preparing model')
    if args.grasp_policy in ['heuristic', 'flingbot', 'pick_and_place']:
        grasp_model = GraspModel(model_type=args.grasp_policy, rotation_num=args.grasp_rotation_num).to(grasp_device)
        if args.grasp_checkpoint is None:
            args.grasp_checkpoint = args.exp
        print(f'==> Loading grasping checkpoint from {args.grasp_checkpoint}')
        if args.grasp_checkpoint.endswith('.pth'):
            checkpoint = torch.load(args.grasp_checkpoint, map_location=grasp_device)
        else:
            checkpoint = torch.load(os.path.join('exp', args.grasp_checkpoint, 'models', 'grasp_latest.pth'), map_location=grasp_device)
        grasp_model.load_state_dict(checkpoint['state_dict'])
        print(f'==> Loaded grasping checkpoint from {args.grasp_checkpoint}')
    else:
        grasp_model=None
    
    if args.blow_policy == 'learn':
        blow_model = BlowModel(action_sample_num=args.blow_action_sample_num, x_range=args.blow_x_range, z_rotation=args.blow_z_rotation, last_action=args.blow_last_action).to(blow_device)
        
        if args.blow_checkpoint is None:
            args.blow_checkpoint = args.exp
        print(f'==> Loading blowing checkpoint from {args.blow_checkpoint}')
        if args.blow_checkpoint.endswith('.pth'):
            checkpoint = torch.load(args.blow_checkpoint, map_location=grasp_device)
        else:
            checkpoint = torch.load(os.path.join('exp', args.blow_checkpoint, 'models', 'blow_latest.pth'), map_location=grasp_device)
        blow_model.load_state_dict(checkpoint['state_dict'])
        print(f'==> Loaded blowing checkpoint from {args.blow_checkpoint}')
    else:
        blow_model = None

    # get line masks
    line_masks = generate_line_masks(args.grasp_rotation_num, args.grasp_resolution)

    data_sequence = collect_data(
            envs, args, line_masks, None,
            grasp_model, grasp_device,  None, 0,
            blow_model, blow_device,  None, 0, real_env=True
        )
    
    # plot cover area
    cover_percentage = [data_sequence[0]['init_cover_percentage'][0]]
    for i in range(args.grasp_step_num):
        cover_percentage.append(data_sequence[i]['cover_percentage'][0])
    plt.plot(np.arange(args.grasp_step_num + 1), cover_percentage, 'o-')
    plt.ylim(0, 1)
    plt.savefig(os.path.join(os.path.join(args.visualization_dir, 'cover_percentage.png')))

    envs[0].terminate()

    # save visualization
    visualization(args, data_sequence, line_masks, args.visualization_dir, 'real_exp')

    # save all data
    pickle.dump(data_sequence, open(os.path.join(args.visualization_dir, 'test_data.pkl'), 'wb'))

if __name__=='__main__':
    parser = argparse.ArgumentParser('Real Exp')
    # exp & dataset
    parser.add_argument('--exp', type=str, default=None, help='exp name')
    parser.add_argument('--task', default='Test_Real', type=str, help='Please ignore this')
    parser.add_argument('--task_num', default=0, type=int, help='Please ignore this')
    parser.add_argument('--suffix', type=str, default='-0', help='suffix name')
    parser.add_argument('--visualization_num', default=1, type=int, help='visualization num')

    # policy
    parser.add_argument('--policy', default='DextAIRity', type=str, choices=['DextAIRity', 'FlingBot', 'Pick_and_Place'], help='type of policy')

    # grasping
    parser.add_argument('--grasp_step_num', default=5, type=int, help='number of grasping steps')
    parser.add_argument('--grasp_rotation_num', default=8, type=int, help='number of arotations')

    parser.add_argument('--grasp_gpu', type=str, default='0', help='grasping policy gpu id')
    parser.add_argument('--grasp_checkpoint', type=str, default=None, help='exp name of grasp policy checkpoint')
    
    # blowing
    parser.add_argument('--blow_step_num', default=4, type=int, help='number of grasping steps')
    parser.add_argument('--blow_action_sample_num', default=64, type=int, help='number of action samples')
    parser.add_argument('--blow_x_range', default=0.1, type=float, help='x range')
    parser.add_argument('--blow_z_rotation', default=-105, type=float, help='z rotation')
    parser.add_argument('--blow_last_action', type=str2bool, nargs='?', const=True, default=False, help="Input last action")
    parser.add_argument('--blow_time', default=150, type=int, help='Please ignore this')
    
    parser.add_argument('--blow_gpu', type=str, default='0', help='blowing policy gpu id')
    parser.add_argument('--blow_checkpoint', type=str, default=None, help='exp name of blow policy checkpoint')

    args = parser.parse_args()

    # parse policy
    if args.policy == 'DextAIRity':
        args.grasp_policy = 'heuristic'
        args.blow_policy = 'learn'
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

    # resolution of input image
    args.grasp_resolution = 256 if args.grasp_policy == 'heuristic' else 64
    args.blow_resolution = 256

    # flingbot parameters
    args.grasp_scale_options = [1, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875]
    args.grasp_angle_options = np.arange(args.grasp_rotation_num) * (2 * np.pi / args.grasp_rotation_num) if args.grasp_policy == 'pick_and_place' else np.arange(args.grasp_rotation_num) * (np.pi / args.grasp_rotation_num)

    main(args)