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

import albumentations as A
import random
import cv2
'''
The code of this class is adapted from Class DatasetFolder
https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#DatasetFolder
'''
ALLOWED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.flo')

def make_dataset(dir, class_to_idx, extensions):
    
    dir = os.path.expanduser(dir)
    
    ## helper functions
    def has_file_allowed_extension(filename, extensions):
        """Checks if a file is an allowed extension.
        Args:
            filename (string): path to a file
            extensions (tuple of strings): extensions to consider (lowercase)
        Returns:
            bool: True if the filename ends with one of given extensions
        """
        return filename.lower().endswith(extensions)
        
    def is_valid_file(x):
        return has_file_allowed_extension(x, ALLOWED_EXTENSIONS)
    
    ## get image list
    image_list = []
    if bool(class_to_idx):
        for target in sorted(class_to_idx.keys()):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in os.walk(d):
                for fname in fnames:
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        item = (fname, path, class_to_idx[target])
                        image_list.append(item)
    else:
        for root, _, fnames in os.walk(dir):
            for fname in fnames:
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = (path, 0)
                    image_list.append(item)
    
    # get sorted image list
    image_list.sort(key = lambda t: t[0])
    sorted_images = []
    for c in image_list: sorted_images.append(c[1:])
    # print(sorted_images)
    
    
    return sorted_images

def loader(path):
    with open(path, 'rb') as f:
        img = io.imread(f)
        return img

def find_classes(dir):
    """
    Finds the class folders in a dataset.
    Args:
        dir (string): Root directory path.
    Returns:
        tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
    Ensures:
        No class is a subdirectory of another.
    """
    if sys.version_info >= (3, 5):
        # Faster and available in Python 3.5 and above
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    else:
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    # classes_count = []
    # for label in classes:
    #     new_dir = os.path.join(dir, label)
    #     classes_count.append(len([name for name in os.listdir(new_dir) if os.path.isfile(os.path.join(new_dir, name))]))
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    # print(classes_count)
    return classes, class_to_idx    

##################################################################

def get_img_list(root_dir, img_ext='.png'):
    ## make datasets
    classes, class_to_idx = find_classes(root_dir)
    img_samples = make_dataset(root_dir, class_to_idx, img_ext)
    if len(img_samples) == 0:
        raise (RuntimeError("Found 0 files in subfolders of: " + self.root_dir + "\n"
                            "Supported extensions are: " + ",".join(extensions)))

    return img_samples, classes, class_to_idx

#### Callables ####
## Transform
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))
        return img


####################################################################################
class ImageFlowFolder():

    # init
    def __init__(self, root_dir, image_folder=None, transform=None, flow_folder=None, img_ext = '.png', flo_ext = '.flo'):
        # path
        self.root_dir = root_dir
        if image_folder is not None:
            self.image_folder = os.path.join(root_dir, image_folder)
        else:
            self.image_folder = root_dir
        

        if flow_folder is not None:
            self.flow_folder = os.path.join(root_dir, flow_folder)
        else:
            self.flow_folder = None
        self.transform = transform
        
        ##
        classes, class_to_idx = find_classes(self.image_folder)
        img_samples = make_dataset(self.image_folder, class_to_idx, img_ext)
        if len(img_samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                "Supported extensions are: " + ",".join(extensions)))
        classes_counts = []
        for label in classes:
            new_dir = os.path.join(self.image_folder, label)
            classes_counts.append(len([name for name in os.listdir(new_dir) if os.path.isfile(os.path.join(new_dir, name))]))
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        # print('------DEBUG-------')
        print(classes_counts)
        ##
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.img_samples = img_samples
        self.num_of_samples = len(self.img_samples)
        self.classes_counts = classes_counts
        
        ##
        if flow_folder is not None:
            flow_samples = make_dataset(self.flow_folder, class_to_idx, flo_ext)
            if not len(img_samples) == len(flow_samples):
                raise (RuntimeError("The number of rgb images should equal to that of the flow images"))
            self.flow_samples = flow_samples
    
    def __len__(self):
        return self.num_of_samples

    def __getitem__(self, idx):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        img_path, img_target = self.img_samples[idx]
        # print(img_path)
        # print(img_target)
        # image = cv2.imread(img_path)
        image = Image.open(img_path)
        # image = image[:1024, :1024]
        if self.transform is None:
            # to tensor
            # image_aug=image.transpose(2, 0, 1) if len(image.shape) > 2 else image.transpose(0,1)
            # return image_aug, img_target, idx
            return image, img_target, idx
        else:
            #augmented = aug(image=image, mask=mask, bboxes=bboxes, category_id=categories)
            # aug_res = self.transform(image=image, mask=None, bboxes=[], category_id=[])
            image_aug = self.transform(image)
            
            # print(aug_res.shape)
            # to tensor
            # image_aug=aug_res['image'].transpose(2, 0, 1) if len(aug_res['image'].shape) > 2 else aug_res['image'].transpose(0,1)
            #image_aug=aug_res.transpose(2, 0, 1) if len(aug_res.shape) > 2 else aug_res.transpose(0,1)
            return image_aug, img_target, idx
            
                
    def get_dir(self, idx):
        img_path, img_target = self.img_samples[idx]
        if self.flow_folder is not None:
           flow_path, flow_target = self.flow_samples[idx]
           return img_path, flow_path
        return img_path
    def get_classes_count(self):
        # print('get_classes_count is called')
        # print(self.classes_counts)
        return self.classes_counts
    def get_data_with_idx(self, indices):
        if self.transform is None:
            raise("no transform")
        else:
            batch_imgs=[]
            batch_targets=[]
            for idx in indices: 
                img_path, img_target = self.img_samples[idx]
                image = Image.open(img_path)
                # aug_res = self.transform(image=image, mask=None, bboxes=[], category_id=[])
                image_aug = self.transform(image)
                # to tensor
                # image_aug=aug_res['image'].transpose(2, 0, 1) if len(aug_res['image'].shape) > 2 else aug_res['image'].transpose(0,1)
                batch_targets.append(img_target)
                batch_imgs.append(np.asarray(image_aug))
            # to torch
            batch_targets = np.asarray(batch_targets)
            batch_targets = torch.from_numpy(batch_targets)
            batch_imgs = torch.FloatTensor(batch_imgs)
            # batch_imgs = np.asarray(batch_imgs)
            # batch_imgs = torch.from_numpy(batch_imgs)
            return batch_imgs, batch_targets

## Load images concated by flow and rgb images
def LoadImageData(data_dir, has_flow=False, is_training=True):

    normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
    )

    
    training_data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])  

    image_datasets = ImageFlowFolder(os.path.join(data_dir, 'train'), 'images', flow_folder=None, transform=training_data_transforms)
    
    num_classes = len(image_datasets.classes)
    
    return image_datasets, num_classes

    

