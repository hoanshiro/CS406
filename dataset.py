import os
import numpy as np
import pandas as pd
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import io
from torchvision.transforms import transforms


class TrafficSignDataset(Dataset):
    def __init__(self, csv_file, imgs_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.imgs_dir = imgs_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        df = self.annotations
        img_name = df.iloc[index, 0]
        img_path = os.path.join(self.imgs_dir, img_name)
        image = cv2.imread(img_path)
        labels = torch.tensor(np.array(df[df['file_name'] == img_name].iloc[:, -1]))
        df_boxes = df[df['file_name'] == img_name].iloc[:, 1:-2]
        boxes = torch.from_numpy(np.array(df_boxes))

        target = {}

        target["boxes"] = boxes
        target["labels"] = labels
        if self.transform:
            image = self.transform(image)
        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


def get_loader(params, cfg):
    # Load Data

    params = {'batch_size': cfg["batch_size"],
              'shuffle': True,
              'pin_memory': True,
              'collate_fn': collate_fn,
              'num_workers': cfg["num_workers"]}
    train_dataset = TrafficSignDataset(
        csv_file=cfg["train_label"],
        imgs_dir=cfg["train_imgs"],
        transform=transforms.ToTensor(),
    )

    val_dataset = TrafficSignDataset(
        csv_file=cfg["val_label"],
        imgs_dir=cfg["val_imgs"],
        transform=transforms.ToTensor(),
    )

    test_dataset = TrafficSignDataset(
        csv_file=cfg["test_label"],
        imgs_dir=cfg["test_imgs"],
        transform=transforms.ToTensor(),
    )
    train_loader = DataLoader(train_dataset, **params)
    val_loader = DataLoader(val_dataset, **params)
    test_loader = DataLoader(test_dataset, **params)
    return train_loader, val_loader, test_loader
