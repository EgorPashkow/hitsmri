import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np

import matplotlib.pyplot as plt

from augs import SyncRandomAffine, SyncToPILImage, SyncColorJitter, SyncRandomVerticalFlip, SyncRandomHorizontalFlip, \
    SyncToTensor
from torchvision.transforms import ToPILImage


def find_srcs(src_dir):
    src_dir = Path(src_dir)
    tifs = src_dir.glob('*/*.tif')
    img_paths = [tif for tif in tifs if '_mask' not in tif.name]

    return img_paths


def train_val_split(img_paths, val_percentage=0.2):
    train_paths, val_paths = [], []

    for path in img_paths:
        dst = train_paths if np.random.rand() > val_percentage else val_paths
        dst.append(path)

    return train_paths, val_paths


class MRIDataset(Dataset):
    def __init__(self, img_paths, transforms=[]):
        mask_paths = [img_path.parent / img_path.name.replace('.tif', '_mask.tif') for img_path in img_paths]

        self.imgs = [plt.imread(path) for path in img_paths]
        self.masks = [plt.imread(path) for path in mask_paths]

        self.transforms = transforms
        self.aug = True

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item):
        img = self.imgs[item]
        mask = self.masks[item]

        if self.aug:
            for transform in self.transforms:
                img, mask = transform(img, mask)

        return img, mask

class MRIDatasetFLAIR(Dataset):
    def __init__(self, img_paths, transforms=[]):
        mask_paths = [img_path.parent / img_path.name.replace('.tif', '_mask.tif') for img_path in img_paths]

        self.imgs = [plt.imread(path) for path in img_paths]
        self.masks = [plt.imread(path) for path in mask_paths]

        self.transforms = transforms
        self.aug = True

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item):
        img = self.imgs[item]
        mask = self.masks[item]

        if self.aug:
            for transform in self.transforms:
                img, mask = transform(img, mask)
        img = img[[1], ...]
        return img, mask
    
def prepare_datasets(src_dir='./data', transforms=[SyncToPILImage(),
        SyncRandomAffine(degrees=360, translate=(0.05, 0.05), scale=(0.8, 1.2)),
        SyncColorJitter(0.2, 0.2, 0.2),
        SyncRandomVerticalFlip(),
        SyncRandomHorizontalFlip(),
        SyncToTensor()]):
    img_paths = find_srcs(src_dir)
    train_paths, val_paths = train_val_split(img_paths)
    
    train_dataset = MRIDataset(train_paths, transforms=transforms)
    val_dataset = MRIDataset(val_paths, transforms=transforms)

    return train_dataset, val_dataset

def prepare_datasets_FLAIR(src_dir='./data', transforms=[SyncToPILImage(),
        SyncRandomAffine(degrees=360, translate=(0.05, 0.05), scale=(0.8, 1.2)),
        SyncColorJitter(0.2, 0.2, 0.2),
        SyncRandomVerticalFlip(),
        SyncRandomHorizontalFlip(),
        SyncToTensor()]):
    img_paths = find_srcs(src_dir)
    train_paths, val_paths = train_val_split(img_paths)
    
    train_dataset = MRIDatasetFLAIR(train_paths, transforms=transforms)
    val_dataset = MRIDatasetFLAIR(val_paths, transforms=transforms)

    return train_dataset, val_dataset

def download_data():
    from pydrive.auth import GoogleAuth
    from pydrive.drive import GoogleDrive
    from google.colab import auth
    from oauth2client.client import GoogleCredentials

    # Authenticate and create the PyDrive client.
    auth.authenticate_user()
    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()
    drive = GoogleDrive(gauth)

    id = '1K1Uha_IfkWkNZmhUsuGskt2vwiuWGyF_'
    downloaded = drive.CreateFile({'id': id})
    downloaded.GetContentFile('MRI.zip')

def save_model(model, name="model.pt"):
    torch.save(model, name)

def load_model(name="model.pt"):
    return torch.load(name)

if __name__ == '__main__':
    train, val = prepare_datasets()

    train.aug = False
    for i in range(len(train)):
        train.aug = False
        img, mask = train[i]

        if mask.sum():
            f, axes = plt.subplots(6, 2, figsize=(7, 25))
            axes[0, 0].imshow(img)
            axes[0, 1].imshow(mask)

            train.aug = True
            for row in range(1, 6):
                img, mask = train[i]
                axes[row, 0].imshow(img)
                axes[row, 1].imshow(mask)

            plt.show()
