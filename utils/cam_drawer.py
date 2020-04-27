from torch.nn import functional as F
import numpy as np
import cv2, torch, os
from skimage.measure import label, regionprops
# from keras import backend as K


def write_text_to_img(img, text, color=(255, 255, 255), org=(30,50), fontScale = 0.5):
    font = cv2.FONT_HERSHEY_SIMPLEX 
    fontScale = fontScale
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
    if boxAArea + boxBArea - interArea == 0:
        return 0
    else:
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

    X = (X + trans_x)*(uni_scale.cpu())
    Y = (Y + trans_y)*(uni_scale.cpu())
    grid = torch.cat((X, Y), dim=-1)
    return F.grid_sample(image, grid)
    
def camforCount(weight_softmax, feature, img_path, theta=None):
    # render the CAM and output
    img = cv2.imread(img_path, -1) ## [H, W, C]
    # print("feature.shape ,", feature.shape)
    # print("weight_softmax.shape ,", weight_softmax.shape)
    # for batch_inner_id,  target in enumerate()
    if theta is not None:
        img_tensor = torch.from_numpy(img).type(torch.FloatTensor).unsqueeze(0)
        img_tensor = img_tensor.permute(0,3,1,2) ## [B, H, W, C] --> [B, C, H, W]
        img_tensor = image_sampler(img_tensor, theta)
        img = np.asarray(img_tensor.permute(0,2,3,1).squeeze(0)) ## [B, H, W, C] --> [H, W, C]
    height, width, _ = img.shape
    CAM = returnCAM(feature, weight_softmax,len(feature.shape)==3)
    CAM = cv2.resize(CAM, (width, height))
    CAM = CAM/255
    count = np.sum(CAM[0:height, 0:width])/(width*height)

    return count


# generate class activation mapping for the top1 prediction
def returnCAM(feature_conv, weight_softmax, feature_cov_is_3=False):
    # generate the class activation maps upsample to 256x256
    # size_upsample = (256, 256)
    if feature_cov_is_3:
        nc, h, w = feature_conv.shape
    else:
        bz, nc, h, w = feature_conv.shape
    # nc, h, w = feature_conv.shape

    # print(weight_softmax.shape)
    cam = weight_softmax.dot(feature_conv.reshape((nc, h*w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    return cam_img

def returnTensorCAMs(features, class_weights):
    B, C, H, W = features.shape
    features = torch.from_numpy(features)
    class_weights = torch.from_numpy(class_weights).unsqueeze(1)
    CAMs = class_weights.bmm(features.flatten(start_dim=2))
    CAMs = CAMs.reshape(B, 1, H, W)
    return CAMs

def convertTensorToImage(X):
    X = X.permute(1,2,0) ## [C, H, W] --> [H, W, C]
    X = X.numpy()
    img = X - np.min(X)
    img = img / np.max(img)
    img = np.uint8(255 * img)
    return img


class CAMDrawer():
    classes = ['C', 'H', 'P']
    # classes = ['C', 'H', 'P','CH','HC','PH']
    def __init__(self, save_folder, img_width=256, img_height=256, device=None, bar=0.8, classes=None):
        if classes is not None:
            self.classes = classes
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device

        print("in cam_drawer.py, line 118 , device is ", self.device )
        self.save_folder =save_folder
        self.bar = bar
        self.W = img_width
        self.H = img_height

    def draw_cam_at_different_scales(
        self, epoch, gt_lbl, img_path, 
        prob_list, weight_softmax_list, feature_list, theta_list, 
        sub_folder=None, GT=None, logit=None):
        
        lvls = len(prob_list)
        assert len(weight_softmax_list) == lvls
        assert len(feature_list) == lvls
        assert len(theta_list)+1 == lvls

        for lvl in range(lvls):
            self.draw_single_cam(
                epoch, gt_lbl, img_path, 
                prob_list[lvl], weight_softmax_list[lvl][gt_lbl,:], feature_list[lvl], 
                lvl=lvl, theta=theta_list, sub_folder = f"scale_{lvl}")



    def draw_multiple_cams_with_zoom_in_box(self, epoch, gt_lbls, img_paths, probs, class_weights, features, thetas, sub_folder=None, GT=None):
        ## make dir
        save_folder = self.save_folder
        if sub_folder is not None:
            save_folder = os.path.join(save_folder, sub_folder)
        save_folder = os.path.join(save_folder, f"epoch{epoch}")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        CAMs = returnTensorCAMs(features, class_weights)

        B = len(img_paths)
        img_size = torch.tensor([[256, 256]]).repeat([B, 1])
        ctrs = ((thetas[:, 0:2] + 1.0)/2.0 * img_size)
        offset = img_size*thetas[:,2:3]/2.0
        top_lefts = (ctrs - offset).type(torch.LongTensor)
        buttom_rights = (ctrs + offset).type(torch.LongTensor)
        ctrs = ctrs.type(torch.LongTensor)
        
        # render the CAM and output
        for idx in range(B):
            # get original image
            img = cv2.imread(img_paths[idx], -1) ## [H, W, C]
            
            height, width, _ = img.shape
            CAM = convertTensorToImage(CAMs[idx])
            CAM = cv2.resize(CAM, (width, height))
            # print(CAM.shape)
            heatmap = cv2.applyColorMap(CAM, cv2.COLORMAP_JET)
            result = heatmap * 0.3 + img * 0.5
            ## write results on image
            write_text_to_img(result, f"prediction: {self.classes[gt_lbls[idx].item()]}", org=(30,50))
            write_text_to_img(result, f"confidence: {probs[idx].item():.2f}", org=(30,65))
            ## generate bbox from CAM 
            prop = generate_bbox(CAM, self.bar)
            bbox_img = cv2.rectangle(result, (prop.bbox[1], prop.bbox[0]), (prop.bbox[3], prop.bbox[2]), (0, 0, 255), 2)
            ## draw zoom-in box
            bbox_img = cv2.rectangle(bbox_img, (top_lefts[idx, 0], top_lefts[idx, 1]), (buttom_rights[idx, 0], buttom_rights[idx, 1]), (0,255,0), 2)
            ## draw zoom-in center
            bbox_img = cv2.rectangle(bbox_img, (ctrs[idx,0]-1,ctrs[idx,1]-1), (ctrs[idx,0]+1, ctrs[idx,1]+1), (0,255,0), 2)
            img_filename = os.path.basename(img_paths[idx])[:-4]
            output_img_path = os.path.join(save_folder, img_filename+'_'+self.classes[gt_lbls[idx].item()]+'_cam.jpg')
            cv2.imwrite(output_img_path, result)
            # print(f"save cam to {output_img_path}")
        return


        
    def camforCountWithZoomIn(self,epoch,
                weight_softmax0,feature0,
                weight_softmax1,feature1,
                weight_softmax2, feature2,
                img_path, lvl=2, theta=None):
        # render the CAM and output
        img = cv2.imread(img_path, -1) ## [H, W, C]
        if isinstance(theta, list):            
            assert lvl == len(theta), f"Error: lvl {lvl} should equal to len(theta): {theta}"
            img_tensor = torch.from_numpy(img).type(torch.FloatTensor).unsqueeze(0)
            img_tensor.detach()
            img_tensor_scale0 = img_tensor.permute(0,3,1,2) ## [B, H, W, C] --> [B, C, H, W]
            img_tensor_scale1 = image_sampler(img_tensor_scale0, theta[0])
            img_tensor_scale2 = image_sampler(img_tensor_scale1, theta[1])

            img_scale0 = np.asarray(img_tensor_scale0.permute(0,2,3,1).squeeze(0)) ## [B, H, W, C] --> [H, W, C]
            img_tensor_scale1 = img_tensor_scale1.detach()
            img_scale1 = np.asarray(img_tensor_scale1.permute(0,2,3,1).squeeze(0)) ## [B, H, W, C] --> [H, W, C]
            img_tensor_scale2 = img_tensor_scale2.detach()
            img_scale2 = np.asarray(img_tensor_scale2.permute(0,2,3,1).squeeze(0)) ## [B, H, W, C] --> [H, W, C]
        
            # print("img_scale0 shape", img_scale0.shape, " img_scale1 shape, ", img_scale1.shape,"img_scale2 shape ",img_scale2.shape)
            
            # img_filename = os.path.basename(img_path)[:-4]
            # sub_folder = 'jiarui-camforcount'
            # save_folder = self.save_folder
            # print("self save_folder is ",self.save_folder)
            # save_folder = os.path.join(save_folder, sub_folder)
            # save_folder = os.path.join(save_folder, f"{img_filename}")
            # print("save folder name is ", save_folder)
            # if not os.path.exists(save_folder):
            #     os.makedirs(save_folder)


            # output_img_path = os.path.join(save_folder, img_filename+'_'+str(epoch))
            # print("out img path is ", output_img_path)
            # cv2.imwrite(f"{output_img_path}-scale0-crop.jpg", img_scale0)
            # cv2.imwrite(f"{output_img_path}-scale1-crop.jpg", img_scale1)
            # cv2.imwrite(f"{output_img_path}-scale2-crop.jpg", img_scale2)

            height, width, _ = img_scale0.shape
            CAM0 = returnCAM(feature0, weight_softmax0, len(feature0.shape)==3)
            CAM0 = cv2.resize(CAM0, (width, height))
            # heatmap0 = cv2.applyColorMap(CAM0, cv2.COLORMAP_JET)
            # result0 = heatmap0 * 0.3 + img_scale0 * 0.5

            CAM1 = returnCAM(feature1, weight_softmax1, len(feature1.shape)==3)
            CAM1 = cv2.resize(CAM1, (width, height))
            # heatmap1 = cv2.applyColorMap(CAM1, cv2.COLORMAP_JET)
            # result1 = heatmap1 * 0.3 + img_scale1 * 0.5
            
            CAM2 = returnCAM(feature2, weight_softmax2, len(feature2.shape)==3)
            CAM2 = cv2.resize(CAM2, (width, height))
            # heatmap2 = cv2.applyColorMap(CAM2, cv2.COLORMAP_JET)
            # result2 = heatmap2 * 0.3 + img_scale2 * 0.5
            
            CAM0_tmp = CAM0/255
            count0_fullsize = np.sum(CAM0_tmp[0:height, 0:width])/(width*height)
            CAM1_tmp = CAM1/255
            count1_fullsize = np.sum(CAM1_tmp[0:height, 0:width])/(width*height)

            # cv2.imwrite(f"{output_img_path}-scale0-cam.jpg", result0)
            # cv2.imwrite(f"{output_img_path}-scale1-cam.jpg", result1)
            # cv2.imwrite(f"{output_img_path}-scale2-cam.jpg", result2)

            # print("theta", theta)
            out_len_01 = 256*theta[0][0][2].item()
            out_center_01_x = (1 + theta[0][0][0].item()) * 128
            out_center_01_y = (1 + theta[0][0][1].item()) * 128   

            out_len_12 = 256*theta[1][0][2].item()
            out_center_12_x = (1 + theta[1][0][0].item()) * 128
            out_center_12_y = (1 + theta[1][0][1].item()) * 128    

            # print("scale 0,", out_len_01, out_center_01_x ,out_center_01_y)
            # print("scale 1,", out_len_12, out_center_12_x ,out_center_12_y)
            upper_01_x =int(out_center_01_x - out_len_01/2)
            upper_01_y =int(out_center_01_y - out_len_01/2)
            lower_01_x =int(out_center_01_x + out_len_01/2)
            lower_01_y =int(out_center_01_y + out_len_01/2)
            
            
            upper_12_x =int(out_center_12_x - out_len_12/2)
            upper_12_y =int(out_center_12_y - out_len_12/2)
            lower_12_x =int(out_center_12_x + out_len_12/2)
            lower_12_y =int(out_center_12_y + out_len_12/2)
            # cv2.rectangle(
            #     img_scale1, 
            #     (int(out_center_12_x - out_len_12/2), int(out_center_12_y - out_len_12/2)), 
            #     (int(out_center_12_x + out_len_12/2), int(out_center_12_y + out_len_12/2)), 
            #     (0, 255, 255), -1) # cropped region by APN
            # print("coordinate ", int(out_center_12_x - out_len_12/2), int(out_center_12_y - out_len_12/2),int(out_center_12_x + out_len_12/2), int(out_center_12_y + out_len_12/2))

            # cv2.rectangle(
            #     CAM0, 
            #     (0,0), 
            #     (width,upper_01_y),
            #     (0, 0, 0), -1) # cropped region by APN
            # cv2.rectangle(
            #     CAM0, 
            #     (0,0), 
            #     (upper_01_x,height),
            #     (0, 0, 0), -1)
            # cv2.rectangle(
            #     CAM0, 
            #     (0,lower_01_y), 
            #     (width,height),
            #     (0, 0, 0), -1)
            # cv2.rectangle(
            #     CAM0, 
            #     (lower_01_x,0), 
            #     (width,height),
            #     (0, 0, 0), -1)     
            # # print("img_scale1 shape", img_scale1.shape)
            # # cv2.imwrite(f"{output_img_path}-scale0-orignWithZoomin.jpg", CAM0)
            # cv2.rectangle(
            #     CAM1, 
            #     (0,0), 
            #     (width,upper_12_y),
            #     (0, 0, 0), -1) # cropped region by APN
            # cv2.rectangle(
            #     CAM1, 
            #     (0,0), 
            #     (upper_12_x,height),
            #     (0, 0, 0), -1)
            # cv2.rectangle(
            #     CAM1, 
            #     (0,lower_12_y), 
            #     (width,height),
            #     (0, 0, 0), -1)
            # cv2.rectangle(
            #     CAM1, 
            #     (lower_12_x,0), 
            #     (width,height),
            #     (0, 0, 0), -1)     
            # print("img_scale1 shape", img_scale1.shape)


            CAM0_masked = CAM0/255
            count0_masked = np.sum(CAM0_masked[upper_01_y:lower_01_y, upper_01_x:lower_01_x])/((lower_01_x-upper_01_x)*(lower_01_y-upper_01_y))
            CAM1_masked = CAM1/255
            count1_masked = np.sum(CAM1_masked[upper_12_y:lower_12_y, upper_12_x:lower_12_x])/((lower_12_x-upper_12_x)*(lower_12_y-upper_12_y))

            # cv2.imwrite(f"{output_img_path}-scale0-orignWithZoomin.jpg", CAM0)
            # cv2.imwrite(f"{output_img_path}-scale1-orignWithZoomin.jpg", CAM1)

            # print('count 0 fullsize:', count0_fullsize, "count 1 fullsize:",count1_fullsize)
            # print('count 0 zoomin:', count0_masked, "count 1 zoomin:",count1_masked)



            return count0_fullsize,count0_masked,count1_fullsize,count1_masked

    def draw_single_cam(self, epoch, gt_lbl, img_path, prob, weight_softmax, feature, lvl=0, theta=None, sub_folder=None, GT=None):
        ## make dir
        save_folder = self.save_folder
        if sub_folder is not None:
            save_folder = os.path.join(save_folder, sub_folder)
        save_folder = os.path.join(save_folder, f"epoch{epoch}")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        # render the CAM and output
        img = cv2.imread(img_path, -1) ## [H, W, C]
        if theta is not None:
            if isinstance(theta, list):
                # print("in draw singl cam, theta is ,", theta)
                
                assert lvl <= len(theta), "Error: lvl should be no greater than len(theta)"
                img_tensor = torch.from_numpy(img).type(torch.FloatTensor).unsqueeze(0)
                img_tensor = img_tensor.permute(0,3,1,2) ## [B, H, W, C] --> [B, C, H, W]
                for l in range(lvl):
                    # print("in draw single cam, theta[l] is ", theta[l])
                    img_tensor = image_sampler(img_tensor, theta[l])
                img = np.asarray(img_tensor.permute(0,2,3,1).squeeze(0)) ## [B, H, W, C] --> [H, W, C]
            else:
                assert lvl == 0, "Error: theta is not a list!"        
                img_tensor = torch.from_numpy(img).type(torch.FloatTensor).unsqueeze(0)
                img_tensor = img_tensor.permute(0,3,1,2) ## [B, H, W, C] --> [B, C, H, W]
                img_tensor = image_sampler(img_tensor, theta)
                img = np.asarray(img_tensor.permute(0,2,3,1).squeeze(0)) ## [B, H, W, C] --> [H, W, C]
            

        height, width, _ = img.shape
        # CAMs = returnCAM(features_blobs[-1], weight_softmax, 0)
        # CAM = cv2.resize(CAMs[0], (width, height))
        CAM = returnCAM(feature, weight_softmax)
        # print(CAM.shape)
        CAM = cv2.resize(CAM, (width, height))
        heatmap = cv2.applyColorMap(CAM, cv2.COLORMAP_JET)

        # get original image
        result = heatmap * 0.3 + img * 0.5
        write_text_to_img(result, f"prediction: {self.classes[gt_lbl]}", org=(30,50))
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
            if theta is None:
                gt_image[y:y+y_len, x:x+x_len] = 1
            else:
                print(theta)
                new_center_x = (1 + theta[0][0] * theta[0][2]) * 128 
                new_center_y = (1 + theta[0][1] * theta[0][2]) * 128
                new_len = 256 * theta[0][2]
                if (new_center_x - new_len/2).item() < 0:
                    new_upper_left_x = 0
                    border_x = (new_center_x - new_len/2).item()
                    x = x - border_x
                else:
                    new_upper_left_x = (new_center_x - new_len/2).item()

                if (new_center_y - new_len/2).item() < 0:
                    new_upper_left_y = 0
                    border_y = (new_center_y - new_len/2).item()
                    y = y - border_y
                else:
                    new_upper_left_y = (new_center_y - new_len/2).item()
                
                if x - new_upper_left_x < 0:
                    x_len = int((x_len + (x - new_upper_left_x))/theta[0][2])
                    x = 0
                else:
                    x = int((x - new_upper_left_x)/theta[0][2])
                    x_len = int(x_len/theta[0][2])

                if y - new_upper_left_y < 0:
                    y_len = int((y_len + (y - new_upper_left_y))/theta[0][2])
                    y = 0
                else:
                    y = int((y - new_upper_left_y)/theta[0][2])
                    y_len = int(y_len/theta[0][2])
                    
                # print(new_center_y)
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
        output_img_path = os.path.join(save_folder, img_filename+'_'+self.classes[gt_lbl[0].item()]+'_cam.jpg')

        cv2.imwrite(output_img_path, result)
        # print(f"save cam to {output_img_path}")
        return

    def draw_single_cam_and_zoom_in_box(
        self, epoch, gt_lbl, img_path, 
        prob, weight_softmax, feature, theta, 
        sub_folder=None, GT=None):
        ## make dir
        save_folder = self.save_folder
        if sub_folder is not None:
            save_folder = os.path.join(save_folder, sub_folder)
        save_folder = os.path.join(save_folder, f"epoch{epoch}")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        # render the CAM and output 
        img = cv2.imread(img_path, -1) ## [H, W, C]               
        height, width, _ = img.shape
        CAM = returnCAM(feature, weight_softmax)
        CAM = cv2.resize(CAM, (width, height))
        heatmap = cv2.applyColorMap(CAM, cv2.COLORMAP_JET)

        ##
        img_size = torch.tensor([height, width])
        ctr = ((theta[0:2] + 1.0)/2.0 * img_size)
        offset = img_size*theta[2]/2.0
        top_left = (ctr - offset).type(torch.LongTensor)
        buttom_right = (ctr + offset).type(torch.LongTensor)
        ctr = ctr.type(torch.LongTensor)

        # get original image
        result = heatmap * 0.3 + img * 0.5
        write_text_to_img(result, f"prediction: {self.classes[gt_lbl]}", org=(30,50))
        write_text_to_img(result, f"confidence: {prob.item():.2f}", org=(30,65))
        
        #generate bbox from CAM 
        prop = generate_bbox(CAM, self.bar)
        bbox_img = cv2.rectangle(result, (prop.bbox[1], prop.bbox[0]), (prop.bbox[3], prop.bbox[2]), (0, 0, 255), 2)
        ## draw zoom-in box
        bbox_img = cv2.rectangle(bbox_img, (top_left[0], top_left[1]), (buttom_right[0], buttom_right[1]), (0,255,0), 2)
        ## draw zoom-in center
        bbox_img = cv2.rectangle(bbox_img, (ctr[0]-1,ctr[1]-1), (ctr[0]+1, ctr[1]+1), (0,255,0), 2)
        
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
            # if theta is not None:
            #     cv2.rectangle(result,(new_upper_left_x,new_upper_left_y),(new_upper_left_x+250, new_upper_left_y+250),(0,255,0),2)
            #     cv2.rectangle(result,(new_center_x,new_center_y),(new_center_x+128, new_center_y+128),(0,255,0),2)
            #     cv2.rectangle(result,(0,0),(0+128, 0+128),(0,255,0),2)


        img_filename = os.path.basename(img_path)[:-4]
        output_img_path = os.path.join(save_folder, img_filename+'_'+self.classes[gt_lbl[0].item()]+'_cam.jpg')

        cv2.imwrite(output_img_path, result)
        # print(f"save cam to {output_img_path}")
        return