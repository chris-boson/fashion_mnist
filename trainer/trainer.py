# @Author: Christfried Focke <christfriedf>
# @Date:   2020-01-29
# @Email:  christfried.focke@gmail.com


import os
import argparse
import json

import numpy as np
from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from tqdm import tqdm, trange
from thop import profile

from trainer.models.cnn import CNN
from trainer.models.cnn_bn import CNNwithBN
from trainer.models.mobilenetv2 import MobileNetV2
from trainer.models.mobilenetv3 import MobileNetV3
from trainer.logger import get_logger

logger = get_logger()

class Trainer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.config = self.args.__dict__.copy()
        logger.info(self.config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self._build_model()
        self.dataset = self._build_dataset()
        self.optimizer = self._build_optimizer()
        self.loss_fct = self._build_loss_fct()
        self.writer = SummaryWriter(self.args.output_dir)

        self.global_step = 0
        self.epoch = 0
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir)
        logger.info('Trainer initialized.')

    def _build_model(self):
        if self.args.model_architecture == 'CNN':
            model = CNN()
        elif self.args.model_architecture == 'CNNwithBN':
            model = CNNwithBN(num_classes=10, conv_type=self.args.conv_type)
        elif self.args.model_architecture == 'MobileNetV2':
            model = MobileNetV2(n_class=10, input_size=32)
        elif self.args.model_architecture == 'MobileNetV3':
            model = MobileNetV3(n_class=10, input_size=32)

        flops, params = profile(model, input_size=(1, 1, 28, 28))
        logger.info('GFlops: %s' % (flops / 1e9))
        logger.info('MParams: %s' % (params / 1e6))
        self.config['GFlops'] = flops / 1e9
        self.config['MParams'] = params / 1e6

        logger.info(model)
        model.to(self.device)
        return model

    def _build_dataset(self):
        train_dataset = FashionMNIST(
            root='resources/data',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor()
            ])
        )
        test_dataset = FashionMNIST(
            root='resources/data',
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor()
            ])
        )
        return train_dataset, test_dataset

    def _build_loss_fct(self):
        if self.args.loss_fct == 'cross_entropy':
            return nn.CrossEntropyLoss()
        elif self.args.loss_fct == 'nll':
            return F.nll_loss

    def _build_optimizer(self):
        return torch.optim.Adam(
            params=self.model.parameters(),
            lr=self.args.lr
        )

    def train(self):
        data_loader = torch.utils.data.DataLoader(
            self.dataset[0],
            shuffle=True,
            batch_size=self.args.train_batch_size,
            num_workers=8
        )
        for epoch in trange(self.args.max_epochs, desc="Epoch"):
            self.model.train()
            train_loss, train_steps = 0., 0
            self.epoch = epoch
            for batch in data_loader:
                images, labels = batch[0].to(self.device), batch[1].to(self.device)
                logits = self.model(images)
                self.optimizer.zero_grad()
                if self.args.loss_fct == 'nll':
                    logits = F.log_softmax(logits, dim=1)
                loss = self.loss_fct(logits, labels)
                loss.backward()
                train_loss += loss.item()
                self.optimizer.step()

                self.global_step += 1
                train_steps += 1
            train_loss /= train_steps
            _, predictions = logits.max(1)
            self._write_summaries('train', predictions, labels, train_loss)
            self.evaluate()

    def evaluate(self):
        self.model.eval()
        data_loader = torch.utils.data.DataLoader(
            self.dataset[1],
            shuffle=False,
            batch_size=self.args.eval_batch_size
        )
        eval_loss, eval_steps = 0., 0
        all_predictions, all_labels = [], []

        for batch in data_loader:
            images, labels = batch[0].to(self.device), batch[1].to(self.device)
            with torch.no_grad():
                logits = self.model(images)
            if self.args.loss_fct == 'nll':
                logits = F.log_softmax(logits, dim=1)
            loss = self.loss_fct(logits, labels)
            eval_loss += loss.mean().item()
            eval_steps += 1

            _, predictions = logits.max(1)
            all_predictions.append(predictions)
            all_labels.append(labels)
        eval_loss /= eval_steps

        self._write_summaries('eval', torch.cat(all_predictions, 0), torch.cat(all_labels, 0), eval_loss)

    def _write_summaries(self, mode, predictions, labels, loss):
        logger.debug("Labels: \n%s" % labels)
        logger.debug("Predictions: \n%s" % predictions)
        results = {
            'mode': mode,
            'accuracy': (labels == predictions).double().mean().item(),
            'loss': loss
        }
        logger.info(json.dumps(results))
        self.writer.add_scalar('%s/loss' % mode, results['loss'], self.global_step)
        self.writer.add_scalar('%s/acc' % mode, results['accuracy'], self.global_step)

    def save_config(self):
        """ Write the configuration file to the output directory. """
        path = os.path.join(self.args.output_dir, 'config.json')
        with open(path, 'w') as f:
            json.dump(self.config, f, indent=4)

    def save_model(self):
        model_to_save = self.model
        path = os.path.join(self.args.output_dir, 'model.bin')
        torch.save(model_to_save, path)
