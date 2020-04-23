
from models.data_loader import make_dataset

class_to_idx = {"C":0, "H":1, "P":2, "CP":3, "HC":4, "PH":5}

sorted_image_class_list = make_dataset('/userhome/30/yfyang/fyp-larger/seq_data/train/images/', class_to_idx =class_to_idx, extensions='png')


with open("train_image_list_sorted_6.txt", "w") as f:
    for item in sorted_image_class_list:
        print(item)
        f.write(item[0])
        f.write("\n")

with open("train_image_label_sorted_6.txt", "w") as f:
    for item in sorted_image_class_list:
        f.write(str(item[1]))
        f.write("\n")
