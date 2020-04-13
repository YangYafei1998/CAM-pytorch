
from models.data_loader import make_dataset

class_to_idx = {"C":0, "H":1, "P":2}

sorted_image_class_list = make_dataset('../fyp_data/val/', class_to_idx =class_to_idx, extensions='png')


with open("val_image_list_sorted.txt", "w") as f:
    for item in sorted_image_class_list:
        f.write(item[0])
        f.write("\n")

with open("val_image_label_sorted.txt", "w") as f:
    for item in sorted_image_class_list:
        f.write(str(item[1]))
        f.write("\n")
