import argparse
import os
import pickle
import shutil

import numpy as np
import torch
import wandb
from tqdm import tqdm
import utils
from model import BagModel

class BagDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, start_id, end_id, input_type):
        self.dataset_path = dataset_path
        self.input_type = input_type
        self.idx_list = list(range(start_id, end_id))

    def __len__(self):
        return len(self.idx_list)
    
    def __getitem__(self, idx):
        data = pickle.load(open(os.path.join(self.dataset_path, f'data-{self.idx_list[idx]}.pkl'), 'rb'))
        color_image = data['observation'][0][100:512+100, 500:512+500].astype(np.float32).transpose([2, 0, 1]) / 255.0
        depth_image = data['observation'][1][100:512+100, 500:512+500].astype(np.float32)[np.newaxis, ...]
        if 'rgb' in self.input_type:
            if 'depth' in self.input_type:
                observation = np.concatenate([color_image, depth_image], axis=0)
            else:
                observation = color_image
        else:
            observation = depth_image

        delta_action = data['delta_action'].astype(np.float32)
        current_action = data['current_action'].astype(np.float32)
        new_action = delta_action + current_action

        inputs = list()
        if 'abs' in self.input_type:
            inputs.extend([new_action[0], new_action[1], new_action[2]])
        if 'curr' in self.input_type:
            inputs.extend([current_action[0], current_action[1], current_action[2]])
        inputs = np.array(inputs, dtype=np.float32)

        target = np.float32(data['open_flag'])
        return observation, inputs, target


def get_input_dim(input_type):
    input_dim, image_dim = 0, 0
    if 'abs' in input_type:
        input_dim += 3
    if 'curr' in input_type:
        input_dim += 3
    if 'depth' in input_type:
        image_dim += 1
    if 'rgb' in input_type:
        image_dim += 3
    return input_dim, image_dim


def main(args):
    # Set wandb
    wandb.init(
        project='bag-opening',
        name=args.exp
    )
    wandb.config.update(args)

    # Print args
    for key in vars(args):
        print('[{0}] = {1}'.format(key, getattr(args, key)))

    exp_path = os.path.join('exp', args.exp)
    utils.mkdir(exp_path)
    model_path = os.path.join('exp', args.exp, 'models')
    utils.mkdir(model_path)

    dataset, dataloader = dict(), dict()
    dataset['train'] = BagDataset(args.dataset, 0, args.train_num, args.input_type)
    dataset['test'] = BagDataset(args.dataset, args.train_num, args.train_num + args.test_num, args.input_type)
    for data_type in ['train', 'test']:
        dataloader[data_type] = torch.utils.data.DataLoader(dataset[data_type], batch_size=args.batch_size, shuffle=data_type=='train', num_workers=args.num_workers)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    input_dim, image_dim = get_input_dim(args.input_type)

    model = BagModel(input_dim=input_dim, image_dim=image_dim, early_fusion=args.early_fusion).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criteria = torch.nn.BCEWithLogitsLoss()
    for epoch in range(args.epoch):
        print(f'==> epoch {epoch + 1}')

        model.train()
        torch.set_grad_enabled(True)
        train_loss, train_acc = 0, 0
        for observation, action, target in tqdm(dataloader['train']):
            pred = model(observation.to(device), action.to(device))
            loss = criteria(pred, target.to(device))
            train_loss += loss.item() * observation.size(0) / args.train_num
        
            pred_tag = torch.round(torch.sigmoid(pred)).cpu()
            train_acc += float((pred_tag == target).sum()) / args.train_num
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        torch.set_grad_enabled(False)
        test_loss, test_acc = 0, 0
        for observation, action, target in tqdm(dataloader['test']):
            pred = model(observation.to(device), action.to(device))
            loss = criteria(pred, target.to(device))
            test_loss += loss.item() * observation.size(0) / args.test_num
            pred_tag = torch.round(torch.sigmoid(pred)).cpu()
            test_acc += float((pred_tag == target).sum()) / args.test_num

        wandb.log({
            'test_loss': test_loss,
            'test_acc': test_acc,
            'train_loss': train_loss,
            'train_acc': train_acc
        })
        print('train_acc = ', train_acc)
        print('test_acc = ', test_acc)

        save_state = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(save_state, os.path.join(model_path, 'bag_latest.pth'))
        shutil.copyfile(
            os.path.join(model_path, 'bag_latest.pth'),
            os.path.join(model_path, 'bag_epoch_%06d.pth' % (epoch + 1))
        )


if __name__=='__main__':
    parser = argparse.ArgumentParser('Train bag opening')
    # exp & dataset
    parser.add_argument('--exp', type=str, default='bag_opening', help='data name')
    parser.add_argument('--dataset', type=str, default='data/bag_opening', help='dataset_num')
    parser.add_argument('--train_num', type=int, default=4000, help='train num')
    parser.add_argument('--test_num', type=int, default=400, help='test num')

    parser.add_argument('--epoch', default=30, type=int, help='number of epoch')
    parser.add_argument('--num_workers', default=10, type=int, help='num_workers of data loader')
    parser.add_argument('--batch_size', default=256, type=int, help='batch size')
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=1e-6, type=float, help='weight decay')
    parser.add_argument('--gpu', type=str, default='0', help='gpu id')
    args = parser.parse_args()

    args.input_type = ['depth', 'abs', 'curr']

    main(args)