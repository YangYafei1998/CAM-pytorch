import os
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from numpy import nan
extensions = ('.jpg', '.jpeg')

def clear_folder(pathIn, exts=None):
    print('clear jpg files')
    if exts is None:
        print(f"remove {extensions}")
        exts = extensions

    files = [
        os.path.join(pathIn, f) for f in os.listdir(pathIn) 
        if os.path.isfile(os.path.join(pathIn, f)) and f.lower().endswith(exts)
        ]

    for f in files:
        os.remove(f)

def draw_fig(path):
    infile = path+'/info.log'

    cls_loss =[nan]*70
    rank_loss = [nan]*70

    cls_loss0 = [None]*70
    cls_loss1 = [None]*70
    cls_loss2 = [None]*70
    rank_loss0 = [None]*70
    rank_loss1 = [None]*70
    cls_acc_0 = [None]*70
    cls_acc_1 = [None]*70

    box_iou_0= [None]*70
    box_iou_1= [None]*70
    pixel_iou_0= [None]*70
    pixel_iou_1= [None]*70
    with open(infile) as f:
        f = f.readlines()

    cls_readable=False 
    rank_readable=False
    test_readable=False
    epoch=0
    for line in f:
        if 'epoch' in line and line.find('epoch')==0:
            print("epoch existing in ,",line.find("epoch"))
            epoch = int(line[line.find('h')+2 : -2])
            print("epoch is ", epoch)
        if 'Classification' in line:
            cls_readable=True
            rank_readable=False
            test_readable=False
        elif 'APN' in line:
            cls_readable=False
            rank_readable=True  
            test_readable=False    
        elif 'Validation' in line:
            cls_readable=False
            rank_readable=False 
            test_readable=True  

# Validation:
# cls_loss: 1.0841777324676514
# cls_acc_0: 88.97242736816406
# cls_acc_1: 86.71678924560547
# rank_loss: 0.2609584927558899
# box_iou_0: 0.03978225883360806
# box_iou_1: 0.03626521081689314
# pixel_iou_0: 0.0923633165234634
# pixel_iou_1: 0.13962954424401597

        if test_readable:
            if 'cls_acc_0:' in line :
                cls_acc_0[epoch] = float(line[line.find(':')+2:-2])
                # print("update cls_loss, ", line)
            elif 'cls_acc_1' in line :
                cls_acc_1[epoch] = (float(line[line.find(':')+2:-2]))
            elif 'box_iou_0' in line :
                box_iou_0[epoch] = (float(line[line.find(':')+2:-2]))
            elif 'box_iou_1' in line :
                box_iou_1[epoch] = (float(line[line.find(':')+2:-2]))
            elif 'pixel_iou_0' in line :
                pixel_iou_0[epoch] = (float(line[line.find(':')+2:-2]))
            elif 'pixel_iou_1' in line :
                pixel_iou_1[epoch] = (float(line[line.find(':')+2:-2]))




        if cls_readable:
            if 'cls_loss:' in line :
                cls_loss[epoch] = float(line[line.find(':')+2:-2])
                print("update cls_loss, ", line)
            elif 'cls_loss0' in line :
                cls_loss0[epoch] = (float(line[line.find(':')+2:-2]))
            elif 'cls_loss1' in line :
                cls_loss1[epoch] = (float(line[line.find(':')+2:-2]))
            elif 'cls_loss2' in line :
                cls_loss2[epoch] = (float(line[line.find(':')+2:-2]))

        if rank_readable:
            if 'rank_loss:' in line:
                # epoch+=1
                rank_loss[epoch]=(float(line[line.find(':')+2:-2]))
            elif 'rank_loss0' in line :
                rank_loss0[epoch]=(float(line[line.find(':')+2:-2]))
            elif 'rank_loss1' in line :
                rank_loss1[epoch]=(float(line[line.find(':')+2:-2]))


    print("cls loss is ", cls_loss)
    print("cls loss is ", cls_loss)
    print("rank loss is ", rank_loss)
    print("cls_acc_0 is ", cls_acc_0)
    print("cls_acc_1 is ", cls_acc_1)
    print("box_iou_0 is ", box_iou_0)
    print("box_iou_1 is ", box_iou_1)
    print("pixel_iou_0 is ", pixel_iou_0)
    print("pixel_iou_1 is ", pixel_iou_1)

 
    plt.clf()
    plt.figure(1)
    print("len of cls_loss", len(cls_loss))
    img_path= path+f'/e_{int(epoch)}_clsrankl.jpg'
    plt.plot(cls_loss,'o', label='cls_loss' )
    plt.plot(rank_loss,'o',label='rank_loss' )
    plt.plot(cls_loss0,'o', label='cls0_loss' )
    plt.plot(cls_loss1,'o',label='cls1_loss' )
    plt.plot(cls_loss2,'o',label='cls2_loss' )
    plt.plot(rank_loss0,'o',label='rank0_loss' )
    plt.plot(rank_loss1,'o',label='rank1_loss' )

    plt.legend(loc='upper right')
    plt.savefig(img_path, dpi=300) 
    plt.clf()


    plt.figure(2)
    img_path= path+f'/e_{int(epoch)}_accuracy.jpg'
    plt.plot(cls_acc_0 , label='cls_acc_0' )
    plt.plot(cls_acc_1 ,label='cls_acc_1' )
    plt.legend(loc='upper right')
    plt.savefig(img_path, dpi=300) 
    plt.clf()


    plt.figure(3)
    img_path= path+f'/e_{int(epoch)}_iou.jpg'
    plt.plot(box_iou_0 , label='box_iou_0' )
    plt.plot(box_iou_1 ,label='box_iou_1' )
    plt.plot(pixel_iou_0 ,label='pixel_iou_0' )
    plt.plot(pixel_iou_1 ,label='pixel_iou_1' )

    plt.legend(loc='upper right')
    plt.savefig(img_path, dpi=300) 




def write_video_from_images(pathIn, videoName):
    frame_array = []
    files = [f for f in os.listdir(pathIn) if os.path.isfile(os.path.join(pathIn, f))]
    # for sorting the file names properly
    files.sort(key=lambda x: x[5:-4])
    files.sort()
    size = (0, 0)
    for filename in files:
        # skip files other than images
        if not filename.lower().endswith(extensions):
            continue

        # print(filename)
        img = cv2.imread(os.path.join(pathIn,filename))
        height, width, layers = img.shape
        size = (width, height)
        #text = filename
        #write_text_to_img(img, text)
        frame_array.append(img)

    print(size)
    print(videoName)
    out = cv2.VideoWriter(videoName, cv2.VideoWriter_fourcc(*'DIVX'), 10, size)
    for i in range(len(frame_array)):
        out.write(frame_array[i])

    out.release()
    print("finishing writing the video")

if __name__ == '__main__':
    draw_fig("log")
    # config_name = 'coherence'
    # timestamp = '2020-04-14_15-06-56'
    # epoch_idx = 10
    # img_folder = os.path.join('saved', config_name, timestamp, 'result', f"epoch{epoch_idx}")
    # videoname = f'video_{config_name}_{timestamp}_epoch{epoch_idx}.avi'
    # write_video_from_images( img_folder, videoname)
