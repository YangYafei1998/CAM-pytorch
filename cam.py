from torchvision import transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2, torch
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

def undesired_objects (image):
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

def generate_bbox (image):
    # img = cv2.imread('input.jpg', 0)
    bar = image.max()*0.8
    print(f"bar: {bar}")
    ret,thresh1 = cv2.threshold(image,bar,255,cv2.THRESH_BINARY)
    result = undesired_objects(thresh1)
    lbl_0 = label(result) 
    props = regionprops(lbl_0)
    for prop in props:
            print('Found bbox', prop.bbox)
            # bbox_img = cv2.rectangle(image, (prop.bbox[1], prop.bbox[0]), (prop.bbox[3], prop.bbox[2]), (255, 0, 0), 2)
            # cv2.imwrite('bbox_img.jpg', bbox_img)
            return prop
def iou_pixel(inputs, target, smooth=1):
    # Numerator Product
    inter = inputs * target
    #Denominator 
    union= inputs + target - (inputs*target)
    ## Sum over all pixels N x C x H x W => N x C

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
# generate class activation mapping for the top1 prediction
def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

def get_cam(net, features_blobs, img_pil, classes, img_name, path, GT=""):
    params = list(net.parameters())
    weight_softmax = np.squeeze(params[-2].data.cpu().numpy())

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    img_tensor = preprocess(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0)).cuda()
    logit = net(img_variable)

    h_x = F.softmax(logit, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)


    # output: the prediction
    for i in range(0, 3):
        line = '{:.3f} -> {}'.format(probs[i], classes[idx[i].item()])
        # print(line)

    CAMs = returnCAM(features_blobs[-1], weight_softmax, [idx[0].item()])
    # CAMs = returnCAM(features_blobs[-1], weight_softmax, [idx[0].item(), idx[1].item(), idx[2].item()])

    # render the CAM and output
    # print('output CAM.jpg for the top1 prediction: %s' % classes[idx[0].item()])
    img = cv2.imread(img_name)
    height, width, _ = img.shape
    CAM = cv2.resize(CAMs[0], (width, height))
    heatmap = cv2.applyColorMap(CAM, cv2.COLORMAP_JET)
    # heatmap1 = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
    # heatmap2 = cv2.applyColorMap(cv2.resize(CAMs[1],(width, height)), cv2.COLORMAP_JET)
    # heatmap3 = cv2.applyColorMap(cv2.resize(CAMs[2],(width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.3 + img * 0.5
    # result1 = heatmap1 * 0.3 + img * 0.5
    # result2 = heatmap2 * 0.3 + img * 0.5
    # result3 = heatmap3 * 0.3 + img * 0.5
    probability_round = round(probs[0].item(), 2)
    # probability_round1 = round(probs[1].item(), 2)
    # probability_round2 = round(probs[2].item(), 2)
    write_text_to_img(result, "prediction: " + classes[idx[0].item()])
    write_text_to_img(result, "confidence: " + str(probability_round), org=(30,65))
    # write_text_to_img(result1, "prediction: " + classes[idx[0].item()])
    # write_text_to_img(result1, "confidence: " + str(probability_round), org=(50,65))
    # write_text_to_img(result2, "prediction: " + classes[idx[1].item()])
    # write_text_to_img(result2, "confidence: " + str(probability_round1), org=(50,65))
    # write_text_to_img(result3, "prediction: " + classes[idx[2].item()])
    # write_text_to_img(result3, "confidence: " + str(probability_round2), org=(50,65))

    img_filename = img_name[41:-4]
    #generate bbox from CAM 
    prop = generate_bbox(CAM)
    bbox_img = cv2.rectangle(result, (prop.bbox[1], prop.bbox[0]), (prop.bbox[3], prop.bbox[2]), (0, 0, 255), 2)
    # cv2.imwrite(path+img_filename+'_'+classes[idx[0].item()]+'_bbox.jpg', bbox_img)

    if GT != "":
        line = GT.split()
        gt_image = np.zeros((256, 256), dtype=np.uint8)
        x = int(line[2])
        y = int(line[3])
        x_len = int(line[4])
        y_len = int(line[5])
        gt_image[y:y+y_len, x:x+x_len] = 1
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # coverted_heatmap = np.where(heatmap==0,1,heatmap) 
        # localization_loss = F.kl_div(torch.from_numpy(coverted_heatmap).float().log().to(device), torch.from_numpy(gt_image).float().to(device))
        # cv2.imwrite(mask_img_path, gt_image)
        coverted_heatmap = torch.from_numpy(CAM).float().to(device) + 1e-20
        coverted_heatmap = coverted_heatmap/torch.sum(coverted_heatmap)
        # coverted_heatmap = F.softmax(coverted_heatmap.view(1, -1)/0.01, dim=1)
        # coverted_heatmap = torch.where(coverted_heatmap==0,torch.tensor(1e-8).to(device),coverted_heatmap)
        coverted_gt = torch.from_numpy(gt_image).float().to(device) + 1e-20
        # coverted_gt = F.softmax(coverted_gt.view(1, -1), dim=1)
        coverted_gt = coverted_gt/torch.sum(coverted_gt)
        # coverted_gt = torch.where(coverted_gt==0,torch.tensor(1e-8).to(device),coverted_gt)
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
    output_img_path = path+img_filename+'_'+classes[idx[0].item()]+'_cam.jpg'
    # output_img_path2 = path+img_filename+'_'+classes[idx[0].item()]+'_cam1.jpg'
    # output_img_path3 = path+img_filename+'_'+classes[idx[0].item()]+'_cam2.jpg'
    cv2.imwrite(output_img_path, result)
    # cv2.imwrite(output_img_path1, result1)
    # cv2.imwrite(output_img_path2, result2)
    # cv2.imwrite(output_img_path3, result3)


def get_localization_loss(net, features_blobs, img_pil, classes, img_name, path, GT=""):
    params = list(net.parameters())
    weight_softmax = np.squeeze(params[-2].data.cpu().numpy())

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    img_tensor = preprocess(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0)).cuda()
    logit = net(img_variable)

    h_x = F.softmax(logit, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)


    # output: the prediction
    for i in range(0, 3):
        line = '{:.3f} -> {}'.format(probs[i], classes[idx[i].item()])
        # print(line)

    CAMs = returnCAM(features_blobs[-1], weight_softmax, [idx[0].item()])
    img = cv2.imread(img_name)
    height, width, _ = img.shape
    CAM = cv2.resize(CAMs[0], (width, height))
    prop = generate_bbox(CAM)

    if GT != "":
        line = GT.split()
        gt_image = np.zeros((256, 256), dtype=np.uint8)
        x = int(line[2])
        y = int(line[3])
        x_len = int(line[4])
        y_len = int(line[5])
        gt_image[y:y+y_len, x:x+x_len] = 1
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        coverted_heatmap = torch.from_numpy(CAM).float().to(device) + 1e-20
        coverted_heatmap = coverted_heatmap/torch.sum(coverted_heatmap)
        coverted_gt = torch.from_numpy(gt_image).float().to(device) + 1e-20
        coverted_gt = coverted_gt/torch.sum(coverted_gt)
        localization_loss = 1000.0*F.kl_div(coverted_heatmap.log(), coverted_gt, reduction='mean')
         # calculate box iou:
        box_iou = iou_box([x, y, x+x_len, y+y_len],[prop.bbox[1], prop.bbox[0], prop.bbox[3], prop.bbox[2]])
        # calculate pixel iou:
        binary_gt = np.zeros((256, 256), dtype=np.uint8)
        binary_gt[y:y+y_len, x:x+x_len] = 1
        binary_cam = CAM/CAM.max()
        pixel_iou = iou_pixel(binary_cam, binary_gt)

        return localization_loss, box_iou, pixel_iou

