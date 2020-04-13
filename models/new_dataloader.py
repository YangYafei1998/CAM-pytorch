# data processing utils

from __future__ import print_function, division
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image

from skimage import io, transform
import PIL
from PIL import Image
import os
import os.path
import sys

from torchvision import datasets, transforms, utils

# import albumentations as A
import random
import cv2


## Load images concated by flow and rgb images
class ImageDataset():
    def __init__(self, image_path_file, image_label_file, is_training=True):
        
        self.is_training = is_training

        image_path_file
        def file_getlines(filepath):
            with open(filepath) as fp:
                line_list = []
                line = fp.readline()
                while line:
                    line_list.append(line.strip())
                    line = fp.readline()
                return line_list

        self.image_files = file_getlines(image_path_file)
        self.image_labels = file_getlines(image_label_file)


        normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
        )

        if is_training:
            self.data_transforms = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.ToTensor(),
                normalize
            ])
        else:
            self.data_transforms = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])

        # # prepare data
        # normalize = transforms.Normalize(
        #         mean=[0.485, 0.456, 0.406],
        #         std=[0.229, 0.224, 0.225]
        # )


        # transform_train = transforms.Compose([
        #     transforms.RandomResizedCrop(224),
        #     # transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     normalize
        # ])
        # transform_test = transforms.Compose([
        #     transforms.Resize(256),
        #     transforms.CenterCrop(224),
        #     transforms.ToTensor(),
        #     normalize
        # ])

        self.num_classes =len(np.unique(self.image_labels))

    def __getitem__(self, idx):
        img_path, img_target = self.image_files[idx], self.image_labels[idx]
        img_target = torch.tensor(int(img_target)).type(torch.LongTensor)
        image = Image.open(img_path)
        if self.data_transforms is None:
            return image, img_target, idx
        else:
            image_aug = self.data_transforms(image)
            return image_aug, img_target, idx

    def get_data_with_idx(self, indices):
        batch_imgs=[]
        batch_targets=[]
        for idx in indices: 
            image_aug, img_target, _ = self.__getitem__(idx)
            batch_targets.append(img_target)
            batch_imgs.append(np.asarray(image_aug))
        
        # to torch
        batch_targets = np.asarray(batch_targets)
        batch_targets = torch.from_numpy(batch_targets)
        batch_imgs = torch.FloatTensor(batch_imgs)
        return batch_imgs, batch_targets

    def __len__(self):
        return len(self.image_files)



