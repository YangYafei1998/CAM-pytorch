from cam import *
from train import *
import torch, os, os.path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from inception import inception_v3
import torchvision.models as models
from PIL import Image
import glob
from models import MobileNetV2, TCLoss, LoadImageData, ResNet
import cv2
import numpy as np
import time
torch.manual_seed(0)


# functions
CAM             = True
USE_CUDA        = True
RESUME          = False
PRETRAINED      = False
time_consistency = True


# hyperparameters
BATCH_SIZE      = 32
IMG_SIZE        = 224
LEARNING_RATE   = 0.001
EPOCH           = 50
MobileNet = True


# prepare data
normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
)

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

# hook the feature extractor
final_conv=''
features_blobs = []
def hook_feature(module, input, output):
    features_blobs.clear()
    # print("hook feature")
    # print(len(features_blobs))
    features_blobs.append(output.data.cpu().numpy())

def generateVideo(inPath, outPath):
    img_array = []
    img_names = []
    for filename in glob.glob(inPath):
        img_names.append(filename)
        # img = cv2.imread(filename)
        # img_array.append(img)
    img_names.sort()
    for filename in img_names:
        img = cv2.imread(filename)
        img_array.append(img)
        height, width, layers = img.shape
        size = (width,height)    
    out = cv2.VideoWriter(outPath,cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

def emptyFolder(path):
    filelist = glob.glob(path)
    for f in filelist:
        os.remove(f)

def generateCam(inPath, net, features_blobs, classes, outPath):
    image_list = []
    for filename in glob.glob(inPath): 
        image_list.append(filename)
    for i in range(len(image_list)):
        im = Image.open(image_list[i])
        get_cam(net, features_blobs, im, classes, image_list[i], outPath)


## datasets
#load training dataset with self-defined dataloader
train_data, num_classes = LoadImageData('/userhome/30/yfyang/fyp_data/')
trainloader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) 
#load testing data with default dataloader
test_data = datasets.ImageFolder('/userhome/30/yfyang/fyp_data/test/images/', transform=transform_test)
testloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print("data size: {} for training".format(len(trainloader.dataset)))
print("data size: {} for testing".format(len(testloader.dataset)))

# class
classes = {0: 'CDRom', 1: 'HardDrive', 2: 'PowerSupply'}

if MobileNet:
    print("BackBone: MobileNetV2")
    net = MobileNetV2(num_classes=3).cuda()
    net._modules.get('features')[-1].register_forward_hook(hook_feature)
else:
    print("BackBone: ResNet18")
    net = ResNet(num_classes=3).cuda()
    net._modules.get('features')[-2].register_forward_hook(hook_feature)

optimizer = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)


# load checkpoint
if RESUME:
    print("===> Resuming from checkpoint.")
    assert os.path.isfile('checkpoint/epoch40-acc97.7443618774414-1586211346.pt'), 'Error: no checkpoint found!'
    net.load_state_dict(torch.load('checkpoint/epoch40-acc97.7443618774414-1586211346.pt'))

criterion = TCLoss(3)

# test and generate CAM video 
if EPOCH == 0:
    test(testloader, net, USE_CUDA, criterion, 0)

for epoch in range (1, EPOCH + 1):
    net = train(trainloader, net, USE_CUDA, epoch, EPOCH + 1, criterion, optimizer, time_consistency)
    test(testloader, net, USE_CUDA, criterion, epoch)

#generate CAM and output videos
if CAM:
    emptyFolder('/userhome/30/yfyang/pytorch-CAM/result/CAM/CPowerSupply/*.jpg')
    emptyFolder('/userhome/30/yfyang/pytorch-CAM/result/CAM/CHardDrive/*.jpg')
    emptyFolder('/userhome/30/yfyang/pytorch-CAM/result/CAM/CCDRom/*.jpg')
    generateCam('/userhome/30/yfyang/fyp_data/test/images/CHardDrive/*.png', net, features_blobs, classes, '/userhome/30/yfyang/pytorch-CAM/result/CAM/')
    generateCam('/userhome/30/yfyang/fyp_data/test/images/CPowerSupply/*.png', net, features_blobs, classes, '/userhome/30/yfyang/pytorch-CAM/result/CAM/')
    generateCam('/userhome/30/yfyang/fyp_data/test/images/CCDRom/*.png', net, features_blobs, classes, '/userhome/30/yfyang/pytorch-CAM/result/CAM/')
    generateVideo('/userhome/30/yfyang/pytorch-CAM/result/CAM/CPowerSupply/*.jpg', f'/userhome/30/yfyang/pytorch-CAM/result/video/CPowerSupply_{int(time.time())}.avi')   
    generateVideo('/userhome/30/yfyang/pytorch-CAM/result/CAM/CHardDrive/*.jpg', f'/userhome/30/yfyang/pytorch-CAM/result/video/CHardDrive_{int(time.time())}.avi')
    generateVideo('/userhome/30/yfyang/pytorch-CAM/result/CAM/CCDRom/*.jpg', f'/userhome/30/yfyang/pytorch-CAM/result/video/CCDRom_{int(time.time())}.avi')


