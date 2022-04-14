import argparse
import os

import numpy as np
import ray
import torch

import wandb
from sim_env import SimEnv
from model import BlowModel, GraspModel
from train_cloth_unfolding import (collect_data, generate_line_masks,
                                   visualization)
from utils import mkdir, str2bool
import pickle


def main(args):
    # Set wandb
    wandb.init(
        project='cloth-unfolding-test-sim',
        name=args.exp + '-' + args.task_name
    )
    wandb.config.update(args)

    # Save arguments
    exp_dir = os.path.join('exp', args.exp)
    mkdir(exp_dir, clean=False)

    # Set directory. e.g. visualization
    args.visualization_dir = os.path.join('exp', args.exp, f'{args.task_name}-visualization')
    mkdir(args.visualization_dir)

    # ray env
    grasp_device = torch.device(f'cuda:{args.grasp_gpu}' if torch.cuda.is_available() else 'cpu')
    blow_device = torch.device(f'cuda:{args.blow_gpu}' if torch.cuda.is_available() else 'cpu')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.pyflex_gpus
    num_pyflex_gpus = len(args.pyflex_gpus.split(','))
    ray.init()
    RaySimEnv = ray.remote(SimEnv).options(num_cpus=1, num_gpus=num_pyflex_gpus/args.env_num)
    envs = [RaySimEnv.remote(
        gui=False,
        wind_life_time=args.wind_life_time,
        large_grasp=False,
        grasp_policy=args.grasp_policy,
        blow_policy=args.blow_policy,
        blow_z_rotation=args.blow_z_rotation
    ) for _ in range(args.env_num)]

    # get camera matrix (intr, pose)
    args.cam_intr, args.cam_pose = ray.get(envs[0].get_camera_matrix.remote())

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

    if args.task_num % args.env_num != 0:
        print(f'[WARNING] env_num can not be divided by task_num!')

    wandb_info = dict()

    wandb_info[f'grasp-cover-percentage/init'] = list()
    for grasp_step in range(args.grasp_step_num):
        wandb_info[f'grasp-cover-percentage/step-{grasp_step}'] = list()
    if args.blow_policy in ['fixed', 'learn']:
        wandb_info[f'blow-cover-percentage/init'] = list()
        for blow_step in range(args.blow_step_num):
            wandb_info[f'blow-cover-percentage/step-{blow_step}'] = list()


    for batch_id in range(args.task_num // args.env_num):
        print(f'==> {batch_id} / {args.task_num // args.env_num}')
        start_id = batch_id * args.env_num
        end_id = (batch_id + 1) * args.env_num

        # collect data
        data_sequence = collect_data(
            envs, args, line_masks, range(start_id, end_id),
            grasp_model, grasp_device,  None, 0,
            blow_model, blow_device,  None, 0
        )

        # visualization
        if batch_id == 0:
            title = f'test-{args.exp}'
            visualization(args, data_sequence, line_masks, args.visualization_dir, title)

        # save all testing file
        pickle.dump(data_sequence, open(os.path.join(args.visualization_dir, f'test_data-{batch_id}.pkl'), 'wb'))

        wandb_info[f'grasp-cover-percentage/init'].append(np.nanmean(data_sequence[0]['init_cover_percentage']))
        for grasp_step in range(args.grasp_step_num):
            wandb_info[f'grasp-cover-percentage/step-{grasp_step}'].append(np.nanmean(data_sequence[grasp_step]['cover_percentage']))
            if args.blow_policy in ['fixed', 'learn']:
                wandb_info[f'blow-cover-percentage/init'].append(np.nanmean(data_sequence[grasp_step][f'blow_init_cover_percentage-0']))
                for blow_step in range(args.blow_step_num):
                    wandb_info[f'blow-cover-percentage/step-{blow_step}'].append(np.nanmean(data_sequence[grasp_step][f'blow_cover_percentage-{blow_step}']))
    
    wandb_info[f'grasp-cover-percentage/init'] = np.nanmean(wandb_info[f'grasp-cover-percentage/init'])
    for grasp_step in range(args.grasp_step_num):
        wandb_info[f'grasp-cover-percentage/step-{grasp_step}'] = np.nanmean(wandb_info[f'grasp-cover-percentage/step-{grasp_step}'])
    if args.blow_policy in ['fixed', 'learn']:
        wandb_info[f'blow-cover-percentage/init'] = np.nanmean(wandb_info[f'blow-cover-percentage/init'])
        for blow_step in range(args.blow_step_num):
            wandb_info[f'blow-cover-percentage/step-{blow_step}'] = np.nanmean(wandb_info[f'blow-cover-percentage/step-{blow_step}'])

    # save detailed testing results in wandb
    # wandb.log(wandb_info)

    data = [[0, wandb_info[f'grasp-cover-percentage/init']]]
    for grasp_step in range(args.grasp_step_num):
        data.append([grasp_step+1, wandb_info[f'grasp-cover-percentage/step-{grasp_step}']])
        print('step=', grasp_step, 'coverage=', wandb_info[f'grasp-cover-percentage/step-{grasp_step}'])
    table = wandb.Table(data=data, columns = ["step", "cover-percentage"])
    wandb.log({"grasp-cover-percentage" : wandb.plot.line(table, "step", "cover-percentage", title="grasp-cover-percentage")})


if __name__=='__main__':
    parser = argparse.ArgumentParser('Grasp')
    # exp & dataset
    parser.add_argument('--exp', type=str, default=None, help='exp name')
    parser.add_argument('--task', default='Test_Large_Rect', type=str, help='init state dataset path')
    parser.add_argument('--task_num', default=200, type=int, help='number of init state')
    parser.add_argument('--visualization_num', default=16, type=int, help='visualization num')

    # sim env
    parser.add_argument('--pyflex_gpus', type=str, default='0,1,2,3,4,5,6,7', help='pyflex gpu ids')
    parser.add_argument('--env_num', default=40, type=int, help='number of environment')
    parser.add_argument('--wind_life_time', default=60, type=int, help='wind life time')

    # policy
    parser.add_argument('--policy', default='DextAIRity', type=str, choices=['DextAIRity', 'DextAIRity_fixed', 'FlingBot', 'FlingBot_plus', 'Pick_and_Place'], help='type of policy')

    # grasping
    parser.add_argument('--grasp_step_num', default=5, type=int, help='number of grasping steps')
    parser.add_argument('--grasp_rotation_num', default=8, type=int, help='number of arotations')

    parser.add_argument('--grasp_gpu', type=str, default='0', help='grasping policy gpu id')
    parser.add_argument('--grasp_checkpoint', type=str, default=None, help='exp name of grasp policy checkpoint')
    
    # blowing
    parser.add_argument('--blow_step_num', default=4, type=int, help='number of grasping steps')
    parser.add_argument('--blow_time', default=150, type=int, help='number of steps of each blowing')
    parser.add_argument('--blow_action_sample_num', default=64, type=int, help='number of action samples')
    parser.add_argument('--blow_x_range', default=0.1, type=float, help='x range')
    parser.add_argument('--blow_z_rotation', default=-95, type=float, help='z rotation')
    parser.add_argument('--blow_last_action', type=str2bool, nargs='?', const=True, default=False, help="Input last action")
    
    parser.add_argument('--blow_gpu', type=str, default='1', help='blowing policy gpu id')
    parser.add_argument('--blow_checkpoint', type=str, default=None, help='exp name of blow policy checkpoint')

    args = parser.parse_args()

    # parse policy
    if args.policy == 'DextAIRity':
        args.grasp_policy = 'heuristic'
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
    args.task_name = args.task
    args.task = os.path.join('data', args.task)

    # resolution of input image
    args.grasp_resolution = 256 if args.grasp_policy == 'heuristic' else 64
    args.blow_resolution = 256

    # flingbot parameters
    args.grasp_scale_options = [1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75]
    args.grasp_angle_options = np.arange(args.grasp_rotation_num) * (np.pi / args.grasp_rotation_num)

    main(args)

