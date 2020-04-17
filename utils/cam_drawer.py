from torch.nn import functional as F
import numpy as np
import cv2, torch, os
from skimage.measure import label, regionprops
# from keras import backend as K


def write_text_to_img(img, text, color=(255, 255, 255), org=(30,50)):
    font = cv2.FONT_HERSHEY_SIMPLEX 
    fontScale = 0.5
    color = (255, 255, 255) 
    thickness = 1
    img = cv2.putText(img, text, org, font,  
                    fontScale, color, thickness, cv2.LINE_AA) 
    return img


def largest_component(image):
    image = image.astype('uint8')
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)
    sizes = stats[:, -1]

    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    img2 = np.zeros(output.shape)
    img2[output == max_label] = 255
    return img2


def generate_bbox(image, bar):
    bar = image.max()*bar
    ret,thresh1 = cv2.threshold(image,bar,255,cv2.THRESH_BINARY)
    result = largest_component(thresh1)
    lbl_0 = label(result) 
    props = regionprops(lbl_0)
    return props[0]
    # for prop in props:
    #         # print('Found bbox', prop.bbox)
    #         # bbox_img = cv2.rectangle(image, (prop.bbox[1], prop.bbox[0]), (prop.bbox[3], prop.bbox[2]), (255, 0, 0), 2)
    #         # cv2.imwrite('bbox_img.jpg', bbox_img)
    #         return prop


def iou_pixel(inputs, target):
    inter = inputs * target
    union= inputs + target - (inputs*target)
    iou = inter.sum()/union.sum()
    return iou


def iou_box(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

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
    

# generate class activation mapping for the top1 prediction
def returnCAM(feature_conv, weight_softmax):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    # print(weight_softmax.shape)
    cam = weight_softmax.dot(feature_conv.reshape((nc, h*w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    return cam_img


class CAMDrawer():
    classes = ['C', 'H', 'P']
    def __init__(self, save_folder, img_width=256, img_height=256, device=None, bar=0.8, classes=None):
        if classes is not None:
            self.classes = classes
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device

        self.save_folder =save_folder
        self.bar = bar
        self.W = img_width
        self.H = img_height
        

    def draw_cam(self, epoch, prob, idx, weight_softmax, feature, img_path, theta=None, sub_folder=None, GT=None, logit=None):
        ## compute softmax probablity
        save_folder = self.save_folder
        if sub_folder is not None:
            save_folder = os.path.join(save_folder, sub_folder)
        save_folder = os.path.join(save_folder, f"epoch{epoch}")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        # h_x = F.softmax(logit, dim=1).data.squeeze()
        # probs, idx = h_x.sort(0, True)
        
        # render the CAM and output
        img = cv2.imread(img_path, -1) ## [H, W, C]
        if theta is not None:
            img_tensor = torch.from_numpy(img).type(torch.FloatTensor).unsqueeze(0)
            img_tensor = img_tensor.permute(0,3,1,2) ## [B, H, W, C] --> [B, C, H, W]
            img_tensor = image_sampler(img_tensor, theta)
            img = np.asarray(img_tensor.permute(0,2,3,1).squeeze(0)) ## [B, H, W, C] --> [H, W, C]

        height, width, _ = img.shape
        # CAMs = returnCAM(features_blobs[-1], weight_softmax, 0)
        # CAM = cv2.resize(CAMs[0], (width, height))
        CAM = returnCAM(feature, weight_softmax)
        CAM = cv2.resize(CAM, (width, height))
        heatmap = cv2.applyColorMap(CAM, cv2.COLORMAP_JET)

        # get original image
        result = heatmap * 0.3 + img * 0.5
        write_text_to_img(result, f"prediction: {self.classes[idx]}", org=(30,50))
        write_text_to_img(result, f"confidence: {prob.item():.2f}", org=(30,65))
        
        #generate bbox from CAM 
        prop = generate_bbox(CAM, self.bar)
        bbox_img = cv2.rectangle(result, (prop.bbox[1], prop.bbox[0]), (prop.bbox[3], prop.bbox[2]), (0, 0, 255), 2)

        if GT is not None:
            line_coord = GT.split()
            gt_image = np.zeros((256, 256), dtype=np.uint8)
            x = int(line_coord[2])
            y = int(line_coord[3])
            x_len = int(line_coord[4])
            y_len = int(line_coord[5])
            gt_image[y:y+y_len, x:x+x_len] = 1

            
            coverted_heatmap = torch.from_numpy(CAM).float() + 1e-20
            coverted_heatmap = coverted_heatmap/torch.sum(coverted_heatmap)
            coverted_gt = torch.from_numpy(gt_image).float() + 1e-20
            coverted_gt = coverted_gt/torch.sum(coverted_gt)
            
            # calculate kl loss
            kl_loss = 1000.0*F.kl_div(coverted_heatmap.log(), coverted_gt, reduction='mean')
            write_text_to_img(result, f"kl loss: {kl_loss:.4f}", org=(30,80))
            
            # calculate box iou:
            box_iou = iou_box([x, y, x+x_len, y+y_len],[prop.bbox[1], prop.bbox[0], prop.bbox[3], prop.bbox[2]])
            write_text_to_img(result, f"box_iou: {box_iou:.4f}", org=(30,95))
            
            # calculate pixel iou:
            binary_gt = np.zeros((256, 256), dtype=np.uint8)
            binary_gt[y:y+y_len, x:x+x_len] = 1
            binary_cam = CAM/CAM.max()
            pixel_iou = iou_pixel(binary_gt, binary_cam)
            write_text_to_img(result, f"pixel_iou: {pixel_iou:.4f}", org=(30,110))

            cv2.rectangle(result,(x,y),(x+x_len,y+y_len),(0,255,0),2)

        img_filename = os.path.basename(img_path)[:-4]
        output_img_path = os.path.join(save_folder, img_filename+'_'+self.classes[idx[0].item()]+'_cam.jpg')

        cv2.imwrite(output_img_path, result)
        # print(f"save cam to {output_img_path}")
        return