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

from albumentations import GaussNoise, MotionBlur, MedianBlur, OneOf, Compose


## Load images concated by flow and rgb images
class ImageDataset():
    def __init__(self, image_path_file, image_label_file, image_localization_file=None, is_training=True, temporal_coherence=False):
        
        self.is_training = is_training
        self.temporal_coherence = temporal_coherence

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
        if image_localization_file is not None:
            self.image_localization = file_getlines(image_localization_file)


        normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
        )

        ## data augmentation
        def image_augmentation(p=1.0):
            return Compose(
                [
                    GaussNoise(var_limit=(10.0,50.0),mean=0,p=0.2),
                    OneOf([
                        MotionBlur(p=0.2),
                        MedianBlur(blur_limit=3, p=0.1),
                        Blur(blur_limit=3, p=0.1),
                    ], p=0.2),
                    CoarseDropout(max_holes=2, min_holes=0, max_height=15, max_width=15, p=0.3)
                ]
            )

        if not is_training:
            self.data_transforms = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])
        elif augmentation:
            self.data_transforms = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomResizedCrop(224),
                image_augmentation(1.0),
                transforms.ToTensor(),
                normalize
            ])
        else:
            self.data_transforms = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomResizedCrop(224),
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

    ## original
    def __getitem__(self, index):
        if self.is_training:
            if self.temporal_coherence:
                return self.__getitem__coherence(index)
            else:
                return self.__getitem__original(index)
        else:
            return self.__getitem__localization(index)

    def __getitem__original(self, idx):
        img_path, img_target = self.image_files[idx], self.image_labels[idx]
        img_target = torch.tensor(int(img_target)).type(torch.LongTensor)
        image = Image.open(img_path)
        if self.data_transforms is None:
            image = transforms.ToTensor(image)
            return image, img_target, idx
        else:
            image_aug = self.data_transforms(image)
            return image_aug, img_target, idx

    ## for temporal coherence
    def __getitem__coherence(self, index):
        
        indices = [index, index, index]
        if index > 0:
            indices[0] = index-1
        if index < len(self.image_files)-1:
            indices[-1] = index+1

        images_aug, img_targets = [], []
        for idx in indices:
            img_path, img_target = self.image_files[idx], self.image_labels[idx]
            img_targets.append(torch.tensor(int(img_target)).type(torch.LongTensor))
            image = Image.open(img_path)
            images_aug.append(self.data_transforms(image))


        img_targets = torch.stack(img_targets, dim=0)
        # print(img_targets.shape)
        images_aug = torch.stack(images_aug, dim=0)
        # print(images_aug.shape)
        # input()
        return images_aug, img_targets, indices

    def __getitem__localization(self, idx):
        img_path, img_target, img_localization = self.image_files[idx], self.image_labels[idx], self.image_localization[idx]
        img_target = torch.tensor(int(img_target)).type(torch.LongTensor)
        image = Image.open(img_path)
        if self.data_transforms is None:
            image = transforms.ToTensor(image)
            return image, img_target, idx, img_localization
        else:
            image_aug = self.data_transforms(image)
            return image_aug, img_target, idx, img_localization
    
    def get_data_with_idx(self, indices):
        batch_imgs=[]
        batch_targets=[]
        for idx in indices: 
            image_aug, img_target, _ = self.__getitem__(idx)
            # batch_targets.append(img_target)
            # batch_imgs.append(np.asarray(image_aug))
            batch_targets.append(torch.LongTensor(img_target))
            batch_imgs.append(torch.FloatTensor(image_aug))
        
        # # to torch
        # batch_targets = np.asarray(batch_targets)
        # batch_targets = torch.from_numpy(batch_targets)
        # batch_imgs = torch.FloatTensor(batch_imgs)
        return batch_imgs, batch_targets

    def get_fname(self, indices):
        batch_fnames=[]
        for idx in indices:
            batch_fnames.append(self.image_files[idx])
        return batch_fnames

    def __len__(self):
        return len(self.image_files)



