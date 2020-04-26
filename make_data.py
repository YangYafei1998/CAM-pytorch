
from models.data_loader import make_dataset

# class_to_idx = {"C":0, "H":1, "P":2, "CP":3, "HC":4, "PH":5}

class_to_idx = {"C":0, "H":1, "P":2}
sorted_image_class_list = make_dataset('data/seq_data/3Ctrain/images/', class_to_idx =class_to_idx, extensions='png')

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
