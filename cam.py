from torchvision import transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2, torch

def write_text_to_img(img, text, color=(255, 255, 255), org=(50,50)):
    font = cv2.FONT_HERSHEY_SIMPLEX 
    fontScale = 0.5
    color = (255, 255, 255) 
    thickness = 1
    img = cv2.putText(img, text, org, font,  
                    fontScale, color, thickness, cv2.LINE_AA) 
    return img


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

def get_cam(net, features_blobs, img_pil, classes, img_name, path):
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
    # CAMs = returnCAM(features_blobs[-1], weight_softmax, [0,1,2])

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
    write_text_to_img(result, "prediction: " + classes[idx[0].item()])
    write_text_to_img(result, "confidence: " + str(probability_round), org=(50,65))
    
    img_filename = img_name[41:-4]
    output_img_path = path+img_filename+'_'+classes[idx[0].item()]+'_cam.jpg'
    # output_img_path1 = path+img_filename+'_'+classes[idx[0].item()]+'_cam0.jpg'
    # output_img_path2 = path+img_filename+'_'+classes[idx[0].item()]+'_cam1.jpg'
    # output_img_path3 = path+img_filename+'_'+classes[idx[0].item()]+'_cam2.jpg'
    cv2.imwrite(output_img_path, result)
    # cv2.imwrite(output_img_path1, result1)
    # cv2.imwrite(output_img_path2, result2)
    # cv2.imwrite(output_img_path3, result3)
