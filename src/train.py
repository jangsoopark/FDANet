from torchvision import transforms
from torch.utils import data

from torch import optim
from torch import nn

from utils import common
from data import preprocess
from data import loader

from model import network
from model import trainer

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


# noinspection PyTypeChecker
def main():
    logging.info('START')

    experiments_path = os.path.join(common.project_root, args.experiments_path)
    config_path = os.path.join(experiments_path, f'{args.config_name}.json')

    config = json.load(open(config_path, mode='r', encoding='utf-8'))
    name = config['name']

    # Model
    num_classes = config.get('num_classes')
    model = network.architecture[config.get('backbone')](pretrained=config.get('pretrained-backbone'), num_classes=num_classes)
    criterion = nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = optim.AdamW(model.parameters(), lr=config.get('learning-rate'), weight_decay=config.get('weight-decay'))
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max' if num_classes == 1 else 'min', patience=2)
    # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[], gamma=0.5)
    scheduler = None if not config.get('learning-rate-scheduler') else optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=config.get('T_0'), T_mult=config.get('T_mult'))

    _trainer = trainer.Trainer(
        name=name, model=model,
        criterion=criterion, optimizer=optimizer, scheduler=scheduler,
        device=config.get('device'),
        experiments_path=experiments_path
    )

    # Dataset
    train_set = load_dataset(
        path=os.path.join(config.get('dataset-path'), 'train'),
        is_train=True,
        batch_size=config.get('batch-size'),
        num_workers=4,
    )
    valid_set = load_dataset(
        path=os.path.join(config.get('dataset-path'), 'val'),
        is_train=False,
        batch_size=config.get('batch-size'),
        num_workers=1,
    )
    test_set = load_dataset(
        path=os.path.join(config.get('dataset-path'), 'test'),
        is_train=False,
        batch_size=config.get('batch-size'),
        num_workers=1,
    )
    _trainer.train(train_set, valid_set, test_set, config.get('epochs'))
    logging.info('FINISH')


if __name__ == '__main__':
    sys.exit(main())
