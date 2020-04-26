
from models.data_loader import make_dataset

# class_to_idx = {"C":0, "H":1, "P":2, "CP":3, "HC":4, "PH":5}

<<<<<<< HEAD
class_to_idx = {"C":0, "H":1, "P":2}
sorted_image_class_list = make_dataset('data/seq_data/3Ctrain/images/', class_to_idx =class_to_idx, extensions='png')
=======
sorted_image_class_list = make_dataset('/userhome/30/yfyang/fyp-larger/seq_data/train/images/', class_to_idx =class_to_idx, extensions='png')
>>>>>>> f33968c0346e64041ceb0bcbf3e9b06ad2f14c05

print("images number ",len(sorted_image_class_list))
with open("data/seq_data/3Ctrain/train_image_list_sorted_3.txt", "w") as f:
    for item in sorted_image_class_list:
        print(item)
        f.write(item[0])
        f.write("\n")

with open("data/seq_data/3Ctrain/train_image_label_sorted_3.txt", "w") as f:
    for item in sorted_image_class_list:
        f.write(str(item[1]))
        f.write("\n")
