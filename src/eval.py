from torchvision import transforms
from torch.utils import data

from torch import optim
from torch import nn

from utils import common
from data import preprocess
from data import loader

from model import network

from tqdm import tqdm
from skimage import io
import numpy as np
import torch

import argparse
import logging
import json
import sys
import os

common.set_random_seed(12321)

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO,
    format='%(asctime)s %(levelname)s %(module)s:%(funcName)s:%(lineno)d %(message)s'
)

parser = argparse.ArgumentParser(description='Feature Difference Attention Module for Change Detection')
parser.add_argument('--experiments-path', default='experiments', type=str, help='')
parser.add_argument('--config-name', default='config/FDAM-LEVIR', type=str, help='')
parser.add_argument('--epoch', default=100, type=int, help='')
args = parser.parse_args()


def load_dataset(path, is_train, batch_size=2, num_workers=1):
    _transform = [preprocess.ToTensor(),]
    if is_train:
        _transform = [
            preprocess.RandomExchange(p=0.5), 
            preprocess.ToTensor(),
            preprocess.RandomHorizontalFlip(p=0.5),
            preprocess.RandomRotate(), 
        ]
    
    _dataset = loader.Dataset(path, is_train, transform=transforms.Compose(_transform))
    return data.DataLoader(
        _dataset, batch_size=batch_size, shuffle=is_train, num_workers=num_workers
    )


@torch.no_grad()
def eval(m, dataset, device, model_path):
    
    result = {'loss': 0, 'precision': 0, 'recall': 0, 'f1-score': 0}
    """       GT
            T   F
        P  T  TP   FP
        F  FN   TN
    """
    result_path = os.path.join(model_path, 'result')
    conf_mat = np.zeros((2, 2))
    m.eval()
    i = 0
    result = []
    for data in tqdm(dataset, ncols=100):
        x1, x2, y = data
        p = m(x1.to(device), x2.to(device))

        _p = torch.sigmoid(p).cpu().numpy()
        _p = np.where(_p > 0.5, 1, 0) # Note: Important for Inference
        
        diff = (y.numpy() - _p).astype(np.int32)

        tp = np.where(np.logical_and(diff == 0, y == 1), 1, 0)
        fp = np.where(np.logical_and(diff < 0, _p == 1), 1, 0)
        fn = np.where(np.logical_and(diff > 0, _p == 0), 1, 0)
        tn = np.where(np.logical_and(diff == 0, y == 0), 1, 0)
        
        _conf_mat = np.array([
            [np.sum(tp), np.sum(fp)],
            [np.sum(fn), np.sum(tn)]
        ])

        result.append({
            'index': i,
            'gt_path': dataset.dataset.labels[i],
            'confusion-matrix': _conf_mat.tolist(),
        })
        
        conf_mat += _conf_mat
        io.imsave(
            os.path.join(result_path, f'predicitons/{i:06d}.png'), 
            _p.transpose(0, 2, 3, 1).squeeze(0),
            plugin='tifffile', check_contrast=False
        )
        i += 1
    
    json.dump(result, open(os.path.join(result_path, 'result.json'), mode='w', encoding='utf-8'))

    precision = conf_mat[0, 0] / np.sum(conf_mat[0, :])
    recall = conf_mat[0, 0] / np.sum(conf_mat[:, 0])
    iou = conf_mat[0, 0] / (conf_mat[0, 0] + conf_mat[0, 1] + conf_mat[1, 0])
    
    return {
        'precision': precision,
        'recall': recall,
        'f1-score': (2 * precision * recall) / (precision + recall),
        'iou': iou,
        'confusion-matrix': conf_mat.tolist(),
    }


# noinspection PyTypeChecker
def main():
    logging.info('START')

    experiments_path = os.path.join(common.project_root, args.experiments_path)
    config_path = os.path.join(experiments_path, f'{args.config_name}.json')

    config = json.load(open(config_path, mode='r', encoding='utf-8'))
    name = config['name']
    device = torch.device(config['device'])
    
    # Model
    model_path = os.path.join(common.project_root, args.experiments_path, 'model', name)
    os.makedirs(os.path.join(model_path, 'result/predictions'), exist_ok=True)
    # print(os.path.join(model_path, f'model-{args.epoch:03d}.pth'))
    num_classes = config.get('num_classes')
    model = network.architecture[config.get('backbone')](pretrained=config.get('pretrained-backbone'), num_classes=num_classes)
    model.load_state_dict(torch.load(os.path.join(model_path, f'model-{args.epoch: 3d}.pth')))
    # model.load_state_dict(torch.load(os.path.join(model_path, f'model-{args.epoch:03d}.pth')))
    model = model.to(device)

    # Dataset
    test_set = load_dataset(
        path=os.path.join(config.get('dataset-path'), 'test'),
        is_train=False,
        batch_size=1,
        num_workers=1,
    )

    result = eval(model, test_set, device, model_path)
    print(result)
    
    logging.info('FINISH')


if __name__ == '__main__':
    sys.exit(main())
