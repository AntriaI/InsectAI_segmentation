import torch
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import random
import numpy as np
from torchvision import transforms


class InsectSegmentationDataset(Dataset):
    """
    For segmentation, the dataset returns (image, mask).
    Geometric augmentations must be applied to both image and mask
    in exactly the same way.
    """
    def __init__(self, file_list, transform=True, image_size=256):
        self.file_list = file_list
        self.transform_flag = transform
        self.image_size = image_size

        self.color_jitter = transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1)

        # ImageNet normalization for pretrained ResNet34 encoder
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path, mask_path = self.file_list[idx]

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask as grayscale
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not read mask: {mask_path}")

        # Resize image and mask
        if self.image_size:
            image = cv2.resize(
                image,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(
                mask,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_NEAREST)

        apply_augmentation = self.transform_flag and random.random() < 0.5

        # Apply same geometric transforms to image and mask
        if apply_augmentation:
            if random.random() < 0.5:
                image = cv2.flip(image, 1)  # horizontal
                mask = cv2.flip(mask, 1)

            if random.random() < 0.5:
                image = cv2.flip(image, 0)  # vertical
                mask = cv2.flip(mask, 0)

        # Convert image to float in [0, 1]
        image = image.astype(np.float32) / 255.0

        # Convert to tensor [C, H, W]
        image = torch.from_numpy(image).permute(2, 0, 1).float()

        # Apply color jitter only to image, not to mask
        if apply_augmentation:
            image = self.color_jitter(image)

        # Normalize using ImageNet stats for pretrained ResNet34 encoder
        image = (image - self.mean) / self.std

        # Convert mask to binary tensor [1, H, W]
        mask = (mask > 0).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        return image, mask


def load_dataset(images_dir, masks_dir):
    data = []

    for file_name in os.listdir(images_dir):
        if file_name.lower().endswith("_image.png"):
            image_path = os.path.join(images_dir, file_name)

            mask_name = file_name.replace("_image.png", "_mask.png")
            mask_path = os.path.join(masks_dir, mask_name)

            if os.path.exists(mask_path):
                data.append((image_path, mask_path))
            else:
                print(f"Warning: mask not found for image {image_path}")

    return data


def split_dataset(data, split=0.8, seed=42):
    """
    Split dataset by date instead of by individual samples.

    Expected filename format:
        20240906-140117-609698_image.png

    The date is the first part before the first '-',
    e.g. 20240906.
    """

    # Collect unique dates
    unique_dates = set()

    for image_path, _ in data:
        file_name = os.path.basename(image_path)
        date_part = file_name.split("-")[0]
        unique_dates.add(date_part)

    unique_dates = list(unique_dates)

    # Shuffle dates, not samples
    random.seed(seed)
    random.shuffle(unique_dates)

    split_idx = int(len(unique_dates) * split)
    train_dates = set(unique_dates[:split_idx])
    val_dates = set(unique_dates[split_idx:])

    # Assign samples based on date
    train_data = []
    val_data = []

    for image_path, mask_path in data:
        file_name = os.path.basename(image_path)
        date_part = file_name.split("-")[0]

        if date_part in train_dates:
            train_data.append((image_path, mask_path))
        else:
            val_data.append((image_path, mask_path))

    return train_data, val_data


def get_dataloader(config):
    images_dir = config.image_dir
    masks_dir = config.mask_dir

    data = load_dataset(images_dir, masks_dir)
    train_data, val_data = split_dataset(data, split=0.8)

    print(f"Training set size: {len(train_data)}, Validation set size: {len(val_data)}")

    # transform = True only for training data (augment only training data)
    train_dataset = InsectSegmentationDataset(
        train_data,
        transform=True,
        image_size=config.training["image_size"])

    val_dataset = InsectSegmentationDataset(
        val_data,
        transform=False,
        image_size=config.training["image_size"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training["batch_size"],
        shuffle=True)

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training["batch_size"],
        shuffle=False)

    return train_loader, val_loader