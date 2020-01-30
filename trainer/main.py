# @Author: Christfried Focke <christfriedf>
# @Date:   2020-01-29
# @Email:  christfried.focke@gmail.com


import argparse
import json
import os
from typing import Dict

import torch

from trainer.trainer import Trainer
from trainer.logger import get_logger

logger = get_logger()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_architecture", type=str, default='CNN',
                        choices=['CNN', 'CNNwithBN', 'MobileNetV2', 'MobileNetV3'], help='Which model architecture to use.')
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to store config, logs, outputs, models etc.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--train_batch_size", type=int, default=100, help="Training batch size.")
    parser.add_argument("--eval_batch_size", type=int, default=100, help="Eval batch size.")
    parser.add_argument("--max_epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--loss_fct", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'nll'], help="Loss function.")
    parser.add_argument("--conv_type", type=str, default='regular',
                        choices=['regular', 'separable'], help="Type of convolution.")



    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    trainer = Trainer(args)
    trainer.save_config()
    trainer.train()
    trainer.save_model()

if __name__ == '__main__':
    main()