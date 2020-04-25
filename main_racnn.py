
import os
import cv2
import numpy as np
import time
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models
from PIL import Image
import glob
from models.new_dataloader import ImageDataset
from models import TCLoss, RACNN, RACNN3Scale

from racnn_trainer import RACNN_Trainer
from racnn3_trainer import RACNN3_Trainer

from utils.config import ConfigParser
from utils.logger import SimpleLogger


def main(config):
    
    ## set seed
    torch.manual_seed(config['seed'])
    
    ARCH_TYPE = config.get('arch', 'RACNN')
    print("NetArch: ", ARCH_TYPE)

    LEARNING_RATE = config['learning_rate']
    WEIGHT_DECAY = config['weight_decay']
    EPOCH = config['max_epoch']

    augmentation = config.get('augmentation', False)
    print('augmentation: ', augmentation)
    
    temporal = config.get('temporal', False)
    print('temporal: ', temporal)
    # ## dataset
    # train_dataset = ImageDataset(
    #     'image_path_folder/train_image_list_sorted.txt', 
    #     'image_path_folder/train_image_label_sorted.txt', 
    #     is_training=True, temporal_coherence=False)
    ## dataset for 6 classes

    train_dataset = ImageDataset(
        'image_path_folder_6/train_image_list_sorted_6.txt', 
        'image_path_folder_6/train_image_label_sorted_6.txt', 
        is_training=True, 
        augmentation=augmentation,
        temporal_coherence=temporal)

    test_dataset = ImageDataset(
        'image_path_folder/test_image_list_sorted.txt', 
        'image_path_folder/test_image_label_sorted.txt', 
        is_training=False)

    ## network
    # print("BackBone: ResNet18")
    # net = ResNet(num_classes=3, pretrained=config.get('pretrained', True))
    
    ## device
    if config['device'] != -1:
        cuda_id = f"cuda:{config['device']}"
        device = torch.device(cuda_id)
    else:
        device = torch.device("cpu")
    print("RACNN with ResNet50")

    if ARCH_TYPE == 'RACNN':
        net = RACNN(num_classes=3, device=device)
    elif ARCH_TYPE == 'RACNN3':
        net = RACNN3Scale(num_classes=3, lvls=3, device=device)
    else:
        raise NotImplementedError


    ## optimizer
    optimizer = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)
    scheduler  = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 55], gamma=0.1)
    
    ## loss
    criterion = TCLoss(num_classes=3)

    ## logger
    logfname = os.path.join(config["log_folder"], "info.log")
    logger = SimpleLogger(logfname, 'debug')

    ## train-test loop
    if ARCH_TYPE == 'RACNN':
        trainer = RACNN_Trainer(net, optimizer, scheduler, criterion, train_dataset, test_dataset, logger, config)
    elif ARCH_TYPE == 'RACNN3':
        trainer = RACNN3_Trainer(net, optimizer, scheduler, criterion, train_dataset, test_dataset, logger, config)
    else:
        raise NotImplementedError

    # trainer.pretrain()
    trainer.train(EPOCH, do_validation=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('-s', '--seed', default=None, help='random seeds')
    parser.add_argument('-b', '--batch_size', default=None, type=int,
                        help='the size of each minibatch')
    parser.add_argument('--max_epoch', default=None, help='max epochs', type=int)
    
    parser.add_argument('--arch', default="RACNN", type=str, 
                        help='Network architecture')
    parser.add_argument('--augmentation', action='store_true')
    parser.add_argument('--temporal', action='store_true')
    

    parser.add_argument('-wd', '--weight_decay', default=None, type=float,
                        help='the size of each minibatch')
    parser.add_argument('-lr', '--learning_rate', default=None, type=float,
                        help='the size of each minibatch')
    parser.add_argument('--interleave', default=None, type=int,
                        help='the size of each minibatch')

    parser.add_argument('--disable_workers', action="store_true")
    parser.add_argument('--comment', help="comments to the session", type=str)
    parser.add_argument('--config', default=None, type=str,
                        help='JSON config path')

    args = parser.parse_args()
    assert args.config is not None, "Please provide the JSON file path containing hyper-parameters for config the network"

    config = ConfigParser(args.config)

    ## allow cmd-line overide
    # if args.resume is not None:
    config.set_content('resume', args.resume)
    
    if args.arch is not None:
        config.set_content('arch', args.arch)
    if args.device is not None:
        config.set_content('device', args.device)
    if args.seed is not None:
        config.set_content('seed', args.seed)
    if args.batch_size is not None:
        config.set_content('batch_size', args.batch_size)
    if args.max_epoch is not None:
        config.set_content('max_epoch', args.max_epoch)
    if args.disable_workers:
        config.set_content('disable_workers', args.disable_workers)
    if args.augmentation:
        config.set_content('augmentation', args.augmentation)
    if args.temporal:
        config.set_content('temporal', args.temporal)

    ## hyperparameters
    if args.weight_decay is not None:
        config.set_content('weight_decay', args.weight_decay)
    if args.learning_rate is not None:
        config.set_content('learning_rate', args.learning_rate)
    if args.interleave is not None:
        config.set_content('interleave', args.interleave)
    
    ##
    if args.comment is not None:
        config.set_content('comment', args.comment)

    main(config=config.get_config_parameters())