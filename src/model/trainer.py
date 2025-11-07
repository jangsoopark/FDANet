from tqdm import tqdm

from torch import nn
import numpy as np
import torch

import logging
import json
import os


class Trainer(object):

    def __init__(
            self, name, model,
            criterion, optimizer, scheduler=None,
            device='cuda:0', experiments_path=None,
            **params
    ):
        # initialize attributes
        self.device = torch.device(device)
        self.name = name
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.experiments_path = experiments_path

        self.use_grad_clip = params.get('use_grad_clip', False)

        self.history = {
            'train': {'loss': [], },
            'val': {'loss': [], 'precision': [], 'recall': [], 'f1-score': [], 'iou': [], 'confusion-matrix': []},
            'test': {'loss': [], 'precision': [], 'recall': [], 'f1-score': [], 'iou': [], 'confusion-matrix': []}
        }

    def optimize(self, x1, x2, y):
        p = self.model(x1.to(self.device), x2.to(self.device))

        loss = self.criterion(p, y.to(self.device))
        self.optimizer.zero_grad()
        loss.backward()
        if self.use_grad_clip:
            nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
        self.optimizer.step()
        return loss.item()

    # noinspection PyTypeChecker,PyArgumentList
    def train(self, train_set, valid_set, test_set, epochs):
        model_path = os.path.join(self.experiments_path, 'model', self.name)
        os.makedirs(model_path, exist_ok=True)
        learning_rate = self.optimizer.param_groups[0]['lr']

        for epoch in range(epochs):
            _loss = []
            self.model.train()
            for data in tqdm(train_set, ncols=100):
                x1, x2, y = data
                _loss.append(self.optimize(x1, x2, y))

            if self.scheduler is not None:
                learning_rate = self.scheduler.get_last_lr()[0]
                # learning_rate = self.optimizer.param_groups[0]['lr']
                self.scheduler.step()

            _valid_result = self.eval(valid_set)
            _test_result = self.eval(test_set)

            self.history['train']['loss'].append(np.mean(_loss))
            self.history['val']['loss'].append(_valid_result['loss'])
            self.history['val']['precision'].append(_valid_result['precision'])
            self.history['val']['recall'].append(_valid_result['recall'])
            self.history['val']['f1-score'].append(_valid_result['f1-score'])
            self.history['val']['iou'].append(_valid_result['iou'])
            self.history['val']['confusion-matrix'].append(_valid_result['confusion-matrix'])
            
            self.history['test']['loss'].append(_test_result['loss'])
            self.history['test']['precision'].append(_test_result['precision'])
            self.history['test']['recall'].append(_test_result['recall'])
            self.history['test']['f1-score'].append(_test_result['f1-score'])
            self.history['test']['iou'].append(_test_result['iou'])
            self.history['test']['confusion-matrix'].append(_valid_result['confusion-matrix'])

            if self.experiments_path is not None:
                torch.save(self.model.state_dict(), os.path.join(model_path, f'model-{epoch + 1:03d}.pth'))
                json.dump(self.history, open(os.path.join(model_path, 'history.json'), mode='w'), ensure_ascii=True)

            logging.info(f'Results')
            logging.info(f'Epoch: {epoch + 1:03d}/{epochs:03d} | learning_rate={learning_rate}')

            _train_loss = self.history['train']['loss'][-1]
            _valid_loss = self.history['val']['loss'][-1]
            _test_loss = self.history['test']['loss'][-1]
            logging.info(f'Train Loss: {_train_loss}| Valid Loss: {_valid_loss}| Test Loss: {_test_loss}')

            _valid_precision = self.history['val']['precision'][-1]
            _test_precision = self.history['test']['precision'][-1]
            logging.info(f'Valid Precision: {_valid_precision}| Test Precision: {_test_precision}')

            _valid_recall = self.history['val']['recall'][-1]
            _test_recall = self.history['test']['recall'][-1]
            logging.info(f'Valid Recall: {_valid_recall}| Test Recall: {_test_recall}')

            _valid_f1 = self.history['val']['f1-score'][-1]
            _test_f1 = self.history['test']['f1-score'][-1]
            logging.info(f'Valid F1-Score: {_valid_f1}| Test F1-Score: {_test_f1}')

            _valid_iou = self.history['val']['iou'][-1]
            _test_iou = self.history['test']['iou'][-1]
            logging.info(f'Valid IOU: {_valid_iou}| Test IOU: {_test_iou}')

            _valid_acc = (_valid_result['confusion-matrix'][0][0] + _valid_result['confusion-matrix'][1][1]) / np.sum(_valid_result['confusion-matrix'])
            _test_acc = (_test_result['confusion-matrix'][0][0] + _test_result['confusion-matrix'][1][1]) / np.sum(_test_result['confusion-matrix'])
            logging.info(f'Valid Accuracy: {_valid_acc}| Test Acc: {_test_acc}')

    @torch.no_grad()
    def eval(self, eval_set):
        result = {'loss': 0, 'precision': 0, 'recall': 0, 'f1-score': 0}
        """       GT
                T   F
         P  T  TP   FP
            F  FN   TN
        """
        _loss = []
        conf_mat = np.zeros((2, 2))
        self.model.eval()
        for data in tqdm(eval_set, ncols=100):
            x1, x2, y = data
            p = self.model(x1.to(self.device), x2.to(self.device))
            _loss.append(self.criterion(p, y.to(self.device)).cpu().item())

            _p = torch.sigmoid(p).cpu().numpy()
            _p = np.where(_p > 0.5, 1, 0) # Note: Important for Inference

            diff = (y.numpy() - _p).astype(np.int32)

            tp = np.where(np.logical_and(diff == 0, y == 1), 1, 0)
            fp = np.where(np.logical_and(diff < 0, _p == 1), 1, 0)
            fn = np.where(np.logical_and(diff > 0, _p == 0), 1, 0)
            tn = np.where(np.logical_and(diff == 0, y == 0), 1, 0)
            
            conf_mat[0, 0] += np.sum(tp)
            conf_mat[0, 1] += np.sum(fp)
            conf_mat[1, 0] += np.sum(fn)
            conf_mat[1, 1] += np.sum(tn)

        precision = conf_mat[0, 0] / np.sum(conf_mat[0, :])
        recall = conf_mat[0, 0] / np.sum(conf_mat[:, 0])
        iou = conf_mat[0, 0] / (conf_mat[0, 0] + conf_mat[0, 1] + conf_mat[1, 0])
        
        return {
            'loss': np.mean(_loss),
            'precision': precision,
            'recall': recall,
            'f1-score': (2 * precision * recall) / (precision + recall),
            'iou': iou,
            'confusion-matrix': conf_mat.tolist(),
        }
