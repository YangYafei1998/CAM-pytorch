from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from skimage import img_as_ubyte


def image_sampler(image, theta, out_w=256, out_h=256):
    B, C, H, W = image.shape
    grid_X, grid_Y = np.meshgrid(np.linspace(-1,1,out_w),np.linspace(-1,1,out_h))
    grid_X = torch.Tensor(grid_X).unsqueeze(0).unsqueeze(3)
    grid_Y = torch.Tensor(grid_Y).unsqueeze(0).unsqueeze(3)

    theta = theta.unsqueeze(2).unsqueeze(3).unsqueeze(4)
    trans_x, trans_y, uni_scale = theta[:, 0, ...], theta[:, 1, ...], theta[:, 2, ...]
 
    ## grid
    X = grid_X.repeat_interleave(B, dim=0)
    Y = grid_Y.repeat_interleave(B, dim=0)
    X = (X + trans_x)*uni_scale
    Y = (Y + trans_y)*uni_scale
    grid = torch.cat((X, Y), dim=-1)
    return F.grid_sample(image, grid)
    

class GridSampler(nn.Module):
    def __init__(self, device, out_h=224, out_w=224):
        super(GridSampler, self).__init__()
        self.out_w, self.out_h = out_w, out_h
        
        self.grid_X, self.grid_Y = np.meshgrid(np.linspace(-1, 1, self.out_w), np.linspace(-1, 1, self.out_h))
        
        self.grid_X = torch.from_numpy(self.grid_X).type(torch.FloatTensor)
        self.grid_X = self.grid_X.unsqueeze(0).unsqueeze(3)
        self.grid_Y = torch.from_numpy(self.grid_Y).type(torch.FloatTensor)
        self.grid_Y = self.grid_Y.unsqueeze(0).unsqueeze(3)
        
        self.grid_X = self.grid_X.to(device).requires_grad_(False)
        self.grid_Y = self.grid_Y.to(device).requires_grad_(False)

    def forward(self, theta):
        ## theta \in [0, 1]
        B = theta.shape[0] ## batch size
        # print(theta.shape)
        theta = theta.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        trans_x, trans_y, uni_scale = theta[:, 0, ...], theta[:, 1, ...], theta[:, 2, ...]
        # print(trans_x.shape)

        X = self.grid_X.repeat_interleave(B, dim=0)
        Y = self.grid_Y.repeat_interleave(B, dim=0)
        X = (X + trans_x)*uni_scale
        Y = (Y + trans_y)*uni_scale
        # print(X.shape)
        return torch.cat((X, Y), dim=-1)

img = cv2.imread('YDXJ0124_test0188.png', -1)

x = 3
y = 34
x_len = 70
y_len = 117
cv2.rectangle(img,(x,y),(x+x_len,y+y_len),(0,255,0),2)
cv2.imwrite('original.png', img)

theta =torch.FloatTensor([[-0.3, -0.3, 0.65]])

new_center_x = (1 + theta[0][0] * theta[0][2]) * 128 
new_center_y = (1 + theta[0][1] * theta[0][2]) * 128
new_len = 256 * theta[0][2]
new_upper_left_x = max((new_center_x - new_len/2).item(), 0)
new_upper_left_y = max((new_center_y - new_len/2).item(), 0)
print(new_upper_left_x)
print(new_upper_left_y)
# cv2.rectangle(img,(int(new_upper_left_x),int(new_upper_left_y)),(int(new_upper_left_x+new_len),int(new_upper_left_y+new_len)),(0,255,0),2)
# cv2.circle(img, (new_center_x,new_center_y), 2, (0,0,255),2)
# cv2.circle(img, (128,128), 2, (0,0,255),2)
# cv2.circle(img, (128,192), 2, (0,0,255),2)

cv2.imwrite('origin.png', img)

if x - new_upper_left_x < 0:
    x = 0
    x_len = int((x_len + (x - new_upper_left_x))/theta[0][2])
else:
    x = int((x - new_upper_left_x)/theta[0][2])
    x_len = int(x_len/theta[0][2])

if y - new_upper_left_y < 0:
    y = 0
    y_len = int((y_len + (y - new_upper_left_y))/theta[0][2])
else:
    y = int((y - new_upper_left_y)/theta[0][2])
    y_len = int(y_len/theta[0][2])
print(x_len)
print(y_len)

img_tensor = torch.from_numpy(img).type(torch.FloatTensor).unsqueeze(0)
img_tensor = img_tensor.permute(0,3,1,2) ## [B, H, W, C] --> [B, C, H, W]
img_tensor = image_sampler(img_tensor, theta)
img = np.asarray(img_tensor.permute(0,2,3,1).squeeze(0)).copy() ## [B, H, W, C] --> [H, W, C]


# img.astype(np.uint8)

cv2.rectangle(img,(x,y),(x+x_len,y+y_len),(0,255,0),2)
cv2.circle(img, (x,y), 2, (0,0,255),2)

# cv2.rectangle(img,(int(x),int(y)),(int(x+x_len),int(y+y_len)),(int(0),int(255),int(0)),2)
cv2.imwrite('result.png', img)

