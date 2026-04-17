import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import os
import cv2
import random
import numpy as np
from torchvision import transforms
from collections import Counter


#datafeeder class
class BinaryInsectDataset(Dataset):
    def __init__(self, root_dir, file_list, transform=True, image_size=256):
        self.root_dir = root_dir
        self.file_list = file_list
        self.transform_flag = transform
        self.image_size = image_size

        self.base_transform = transforms.ToTensor()

        self.augment = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path, label = self.file_list[idx]

        image = cv2.imread(img_path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.image_size:
            image = cv2.resize(image, (self.image_size, self.image_size))

        #normalize
        image = image.astype(np.float32)
        image = (image - image.mean()) / (image.std() + 1e-6)

        image = self.base_transform(image)

        #apply augmentation only for training dataset
        if self.transform_flag and random.random() < 0.5:
            image = self.augment(image)

        label = torch.tensor([label], dtype=torch.float32)

        return image, label

#load data from given dir [ dir/insect, dir/no-insect]
def load_dataset(root_dir):
    data = []
    classes = {
        "insect": 1,
        "no-insect": 0
    }
    for class_name, label in classes.items():
        print(root_dir, class_name)
        class_path = os.path.join(root_dir, class_name)
        for file in os.listdir(class_path):
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                data.append((os.path.join(class_path, file), label))

    return data


#train/val split, default 80-20
def split_dataset(data, split=0.8):
    random.shuffle(data)
    split_idx = int(len(data) * split)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    return train_data, val_data



#main dataloader function to be called from run.py
def get_dataloader(config):

    data = load_dataset(config.data_path)
    train_data, val_data = split_dataset(data, split=0.8)

    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    train_dataset = BinaryInsectDataset(config.data_path, train_data, transform=True)
    val_dataset = BinaryInsectDataset(config.data_path, val_data, transform=False)


    train_loader = DataLoader(train_dataset, batch_size=config.training['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.training['batch_size'], shuffle=False)

    return train_loader, val_loader


