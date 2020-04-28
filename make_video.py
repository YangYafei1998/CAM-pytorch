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

    with open(infile) as f:
        f = f.readlines()

    cls_readable=False 
    rank_readable=False
    epoch=0
    for line in f:
        if 'epoch' in line:
            epoch = int(line[line.find('h')+2 : -2])
            print("epoch is ", epoch)
        if 'Classification' in line:
            cls_readable=True
            rank_readable=False
        elif 'APN' in line:
            cls_readable=False
            rank_readable=True      
        elif 'Validation' in line:
            cls_readable=False
            rank_readable=False   

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
    draw_fig("saved/racnn_coherence/2020-04-26_17-09-19/log")
    # config_name = 'coherence'
    # timestamp = '2020-04-14_15-06-56'
    # epoch_idx = 10
    # img_folder = os.path.join('saved', config_name, timestamp, 'result', f"epoch{epoch_idx}")
    # videoname = f'video_{config_name}_{timestamp}_epoch{epoch_idx}.avi'
    # write_video_from_images( img_folder, videoname)
