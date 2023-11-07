import torch
import torch.nn as nn
import torch.optim as optim 
import torchvision
from torchvision import models
from torchvision import transforms
import torchvision.models as models
from tqdm import tqdm





def get_model(config):
    if str(config.get('TRAIN_OPTIONS', 'dataset')) == 'CelebA_HQ_face_gender_dataset':
        num_of_class =2
    else:
        num_of_class =307

    # Changing number of model's output classes to 1
    #for resnet18
    if str(config.get('TRAIN_OPTIONS', 'dataset')) == 'densenet121':
        model = models.densenet121(pretrained=True)
        num_features = model.classifier.in_features
        model.fc = nn.Linear(num_features, num_of_class) 

        

    #for resnet50
    elif str(config.get('TRAIN_OPTIONS', 'dataset')) == 'mnasnet':
        model = models.mnasnet1_0(pretrained=True)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_of_class)

       

    # for densenet 121
    elif str(config.get('TRAIN_OPTIONS', 'dataset')) == 'resnet101':
        model = models.resnet101(pretrained=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_of_class)

        
    #for vgg19_bn
    elif str(config.get('TRAIN_OPTIONS', 'dataset')) == 'resnet18':
        model = models.resnet18(pretrained=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_of_class) 


        

    # Transfer execution to GPU
    model = model.to('cuda')
    
    return model 