import os
import random

def data_split(path):
    # Path to the dataset
    path_to_dataset = f"{path}/JPEGImages"

    list_of_images = os.listdir(path_to_dataset) # assuming that every image has a label
    shuffled_list = random.shuffle(list_of_images) # shuffle the list of images

    # split the list into 80% train, 10% test and 10% validation 
    train = list_of_images[:int(len(list_of_images)*0.8)]
    test = list_of_images[int(len(list_of_images)*0.8):int(len(list_of_images)*0.9)]
    validation = list_of_images[int(len(list_of_images)*0.9):]

    # write the lists to a file
    with open(f"{path}/train.txt", "w") as f:
        for item in train:
            f.write("%s\n" % item)

    with open(f"{path}/training_range.txt", "w") as f:
        for item in train:
            f.write(f"{item.split('.')[0]}\n")

    with open(f"{path}/test.txt", "w") as f:
        for item in test:
            f.write("%s\n" % item)

    with open(f"{path}/validation.txt", "w") as f:
        for item in validation:
            f.write("%s\n" % item)

if __name__ == "__main__":
    data_split(path)