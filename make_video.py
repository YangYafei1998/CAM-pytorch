import os
import cv2
import numpy as np
import glob

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
    # print(len(files))
    size = (0, 0)
    for filename in files:

        # skip files other than images
        # if not filename.lower().endswith(extensions):
        #     continue

        # print(filename)
        img = cv2.imread(os.path.join(pathIn,filename))
        height, width, layers = img.shape
        size = (width, height)
        #text = filename
        #write_text_to_img(img, text)
        frame_array.append(img)

    print(size)
    print(videoName)
    # print(len(frame_array))
    out = cv2.VideoWriter(videoName, cv2.VideoWriter_fourcc(*'DIVX'), 10, size)
    for i in range(len(frame_array)):
        out.write(frame_array[i])

    out.release()
    print("finishing writing the video")

if __name__ == '__main__':
    config_name = 'coherence'
    timestamp = '2020-04-14_15-06-56'
    epoch_idx = 10
    img_folder = os.path.join('saved', config_name, timestamp, 'result', f"epoch{epoch_idx}")
    videoname = f'video_{config_name}_{timestamp}_epoch{epoch_idx}.avi'
    write_video_from_images('/userhome/30/yfyang/fyp-larger/seq_data/train/images/CP', 'traing-data-CP.avi')
