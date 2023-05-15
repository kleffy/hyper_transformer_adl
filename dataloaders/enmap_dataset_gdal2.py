import os
from osgeo import gdal
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class PatchDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['filename']
        x1 = row['x']
        y1 = row['y']
        patch_size = row['patch_size']

        img_ds = gdal.Open(img_path)
        # img_ds = img_ds.GetRasterBand(1)
        patch = img_ds.ReadAsArray(int(x1 + (x1 % patch_size)), int(y1 + (y1 % patch_size)), patch_size, patch_size)

        if self.transform:
            patch = self.transform(patch)

        return patch

if __name__ == "__main__":
    pass
