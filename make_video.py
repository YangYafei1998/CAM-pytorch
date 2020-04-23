import os
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

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

def draw_fig(path):
    infile = path+'/info.log'

    cls_loss =[]
    rank_loss = []

    with open(infile) as f:
        f = f.readlines()

    readable=False 
    epoch=0
    for line in f:
        if 'Classification' in line or 'APN' in line:
            readable=True
        if 'Validation' in line:
            readable=False
        if 'cls_loss' in line and readable:
            epoch+=1
            cls_loss.append(float(line[line.find(':')+2:-2]))
        if 'rank_loss' in line and readable:
            epoch+=1
            rank_loss.append(float(line[line.find(':')+2:-2]))

    plt.clf()
    plt.figure(1)
    print("len of cls_loss", len(cls_loss))
    img_path= path+f'/e_{int(epoch/4)}_clsrankl.jpg'
    plt.plot(cls_loss,label='cls_loss' )
    plt.plot(rank_loss,label='rank_loss' )
    plt.legend(loc='upper right')
    plt.savefig(img_path, dpi=300) 
    plt.clf()

if __name__ == '__main__':
    draw_fig("saved/racnn_coherence/2020-04-23_02-27-06racnn/log")
    # config_name = 'coherence'
    # timestamp = '2020-04-14_15-06-56'
    # epoch_idx = 10
    # img_folder = os.path.join('saved', config_name, timestamp, 'result', f"epoch{epoch_idx}")
    # videoname = f'video_{config_name}_{timestamp}_epoch{epoch_idx}.avi'
    # write_video_from_images( img_folder, videoname)
