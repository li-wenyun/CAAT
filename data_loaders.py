import os
from PIL import Image
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms




transforms_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(), # data augmentation
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # normalization
])

transforms_test = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def get_dataloader(config):
    data_dir =str(config.get('DATA_PATHS', 'data_dir'))
    data_dir=data_dir+str(config.get('TRAIN_OPTIONS', 'dataset'))+'/'
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transforms_train)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transforms_test)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=int(config.get('TRAIN_OPTIONS', 'batch_size')), 
                                                   shuffle=True, num_workers=int(config.get('TRAIN_OPTIONS', 'num_workers')))
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=int(config.get('TRAIN_OPTIONS', 'batch_size')), 
                                                  shuffle=False, num_workers=int(config.get('TRAIN_OPTIONS', 'num_workers')))
    return train_dataloader,test_dataloader,train_dataset,test_dataset
    