"""Importing necessary packages"""

import numpy as np
import numpy as np
import pandas as pd

import os

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc
from matplotlib.ticker import MaxNLocator
from pylab import rcParams

import torch
import torchvision
import PIL

from torch import nn, optim
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader,SubsetRandomSampler
from torchvision import models,datasets
from torch.optim import lr_scheduler

from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
from pytorch_model_summary import summary

import random

import cv2
import PIL.Image as Image

from google.colab.patches import cv2_imshow

from sklearn.metrics import confusion_matrix, classification_report

from collections import defaultdict

from utils import train_epoch,train_model,eval_model,plot_training_history,visualize_images

"""Mounting the drive"""

from google.colab import drive 
drive.mount('/content/drive')
root_dir_path = os.path.join(os.getcwd(),'drive','MyDrive','Labelled Faces In The Wild Dataset')

"""Dataset Paths"""

original_dir_path = os.path.join(root_dir_path,'lfw-deepfunneled','lfw-deepfunneled')
original_cropped_dir_path = os.path.join(root_dir_path,'lfw-deepfunneled','lfw-deepfunneled_cropped')


surgical_masked_dir_path = os.path.join(root_dir_path,'lfw-deepfunneled','lfw-deepfunneled_masked_surgical')
surgical_masked_cropped_dir_path = os.path.join(root_dir_path,'lfw-deepfunneled','lfw-deepfunneled_masked_surgical_cropped')


KN95_masked_dir_path = os.path.join(root_dir_path,'lfw-deepfunneled','lfw-deepfunneled_masked_KN95')
KN95_masked_cropped_dir_path = os.path.join(root_dir_path,'lfw-deepfunneled','lfw-deepfunneled_masked_KN95_cropped')

"""Check for gpu availability"""

if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

"""Configuring some settings for visualization and seeding the values to makes the results as much reproducible as possible"""

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
# %config InlineBackend.figure_format='retina'

sns.set(style='darkgrid', palette='muted', font_scale=1.2)

colors_pallette = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]

sns.set_palette(sns.color_palette(colors_pallette))

rcParams['figure.figsize'] = 16, 12

seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

"""Helper Functions for training , validation and visualizations"""

"""Visualize images from the directories"""

visualize_images(path = original_cropped_dir_path,n_samples= 1)
visualize_images(path = surgical_masked_cropped_dir_path , n_samples= 1)
visualize_images(path = KN95_masked_cropped_dir_path , n_samples= 1)

"""Preparing the dataset and dataloaders"""

batch_size = 64
validation_size = .2
test_size = 0.1
shuffle_dataset = True
data_dir = surgical_masked_cropped_dir_path
num_workers = 0 if os.name == 'nt' else 8

transformations  = T.Compose([
    T.Resize(size = 256),
    T.ToTensor(),
    fixed_image_standardization
])

dataset = datasets.ImageFolder(data_dir, transform=transformations)


dataset_size = len(dataset)

indices = np.arange(dataset_size)

test_split = int(np.floor(test_size*dataset_size))

train_indices = indices[test_split:]
test_indices = indices[:test_split]


# Creating data indices for training and validation splits:
train_val_dataset_size = len(train_indices)
train_val_indices = list(range(train_val_dataset_size))

val_split = int(np.floor(validation_size * train_val_dataset_size))

if shuffle_dataset :
    np.random.seed(seed_val)
    np.random.shuffle(train_val_indices)

train_indices, val_indices = train_val_indices[val_split:], train_val_indices[:val_split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler(test_indices)

train_loader = torch.utils.data.DataLoader(
                                dataset, batch_size=batch_size, 
                                sampler=train_sampler,drop_last = True,
                                num_workers = num_workers
                )
validation_loader = torch.utils.data.DataLoader(
                                dataset, batch_size=batch_size,
                                sampler=valid_sampler,drop_last = True,
                                num_workers = num_workers
                    )

test_loader = torch.utils.data.DataLoader(
                                dataset, batch_size=batch_size,
                                sampler=valid_sampler,drop_last = True,
                                num_workers = num_workers
                    )

print("Total number of different identies in the dataset {}".format(len(dataset.class_to_idx)))

print("Total number of samples in the dataset {}".format(len(dataset.imgs)))

"""Defining our model"""

model = InceptionResnetV1(
    classify=True,
    pretrained='vggface2',   ### 'casia-webface can also be used'
    num_classes=len(dataset.class_to_idx)
).to(device)

print(model)

"""Keras type Model Summary"""

print(summary(model, torch.zeros((1, 3, 256, 256)).to(device), show_input=True))

"""Setting the optimizer,scheduler and loss criterion for the model training"""

logits = model.logits.parameters()          ## optimizing only the last layers
optimizer = optim.Adam(logits, lr=0.001)        
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)  ## setting up the learning rate scheduler
criterion = nn.CrossEntropyLoss()         

## training the model
model , history = train_model(
    model = model , train_data_loader = train_loader , val_data_loader = validation_loader,train_dataset_size = len(train_indices),
    val_dataset_size = len(val_indices) , optimizer = optimizer , criterion = criterion , scheduler = scheduler , 
    device = device , n_epochs = 20
)

"""Saving the model """

torch.save( model.state_dict() , os.path.join(root_dir_path ,"Models" ,"model_name") )

plot_training_history(history)

""" Loading the model from saved checkpoint"""
state_dict = torch.load(os.path.join(root_dir_path,'Models','model_name'))
model.load_state_dict(state_dict=state_dict)

## evaluating the model on test dataset
print(eval_model(model = model , data_loader= test_loader , criterion=criterion,device=device , n_examples=len(test_indices)))


optimizer = optim.Adam(model.parameters(), lr=0.001)  ## optimizing the entire model
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3) ## setting up the learning rate scheduler
criterion = nn.CrossEntropyLoss()

## training the model
model , history = train_model(
    model = model , train_data_loader = train_loader , val_data_loader = validation_loader,train_dataset_size = len(train_indices),
    val_dataset_size = len(val_indices) , optimizer = optimizer , criterion = criterion , scheduler = scheduler , 
    device = device , n_epochs = 20
)

torch.save( model.state_dict() , os.path.join(root_dir_path ,"Models" ,"model_name") )

plot_training_history(history)

state_dict = torch.load(os.path.join(root_dir_path,'Models','model_name'))
model.load_state_dict(state_dict=state_dict)

print(eval_model(model = model , data_loader= test_loader , criterion=criterion,device=device , n_examples=len(test_indices)))

