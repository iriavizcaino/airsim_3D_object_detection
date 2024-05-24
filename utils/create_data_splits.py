import os
import random

# Path to the dataset
path_to_dataset = r"/home/ivizcaino/workspace/airsim_3D_object_detection/Fire_Extinguisher/JPEGImages" 

list_of_images = os.listdir(path_to_dataset) # assuming that every image has a label
shuffled_list = random.shuffle(list_of_images) # shuffle the list of images

# split the list into 80% train, 10% test and 10% validation 
train = list_of_images[:int(len(list_of_images)*0.8)]
test = list_of_images[int(len(list_of_images)*0.8):int(len(list_of_images)*0.9)]
validation = list_of_images[int(len(list_of_images)*0.9):]

# write the lists to a file
with open("train.txt", "w") as f:
    for item in train:
        f.write("%s\n" % item)

with open("training_range.txt", "w") as f:
    for item in train:
        f.write(f"{item.split('.')[0]}\n")

with open("test.txt", "w") as f:
    for item in test:
        f.write("%s\n" % item)

with open("validation.txt", "w") as f:
    for item in validation:
        f.write("%s\n" % item)