from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data.dataset import Dataset
import os
from PIL import Image




class CelebADataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, f"{idx+1:06d}.jpg")
        image = Image.open(img_name)
        
        if self.transform:
            image = self.transform(image)
        
        return image

class LFWDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, f"{idx+1:06d}.jpg")
        image = Image.open(img_name)
        
        if self.transform:
            image = self.transform(image)
        
        return image

class LFWDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset=LFWDataset(self.data_dir,transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class CelebDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset=CelebADataset(root_dir=self.data_dir,transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
