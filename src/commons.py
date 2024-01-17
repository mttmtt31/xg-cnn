import random
from torch.utils.data import Subset
import torch
from .dataset import HeatmapDataset
import datetime
from .models.xg_cnn import XGCNN


def train_val_split(dataset, train_size:float=0.8):
    random.seed(100)
        
    # Perform train/val split (on the original set)
    augmentation = dataset.augmentation
    all_indices = list(range(len(dataset) if not augmentation else int(len(dataset) / 2)))
    # Randomly split the remaining elements
    random.shuffle(all_indices)
    split_point = int(len(all_indices) * train_size)
    # divide train and validation indices (with augmentation if needed)
    if augmentation:
        train_indices = all_indices[:split_point] + [idx + dataset.real_length for idx in all_indices[:split_point]]
    else:
        train_indices = all_indices[:split_point]
    val_indices = all_indices[split_point:]
    # subset dataset
    train_dataset = Subset(dataset, indices = train_indices)
    val_dataset = Subset(dataset, indices = val_indices)

    return train_dataset, val_dataset

def load_model(dropout:float=0.0):
    # Load the XGCNN model
    model = XGCNN(dropout=dropout)

    return model

def load_heatmap_dataset(data_path, labels_path, augmentation, g_filter):
    # Load the dataset
    return HeatmapDataset(data_path=data_path, labels_path=labels_path, augmentation=augmentation, g_filter=g_filter)

def set_optimiser(model, optim, learning_rate, weight_decay):
    if optim.lower() == 'adam': 
        optimiser = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay=weight_decay)  
    elif optim.lower() == 'adamw':
        optimiser = torch.optim.AdamW(model.parameters(), lr = learning_rate, weight_decay=weight_decay) 
    elif optim.lower() == 'sgd':
        optimiser = torch.optim.SGD(model.parameters(), lr = learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f'Specified optimiser {optim} not implemented. Should be one of ["adam", "adamw", "sgd"]')

    return optimiser

def save_model(model):
    now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    torch.save(model.state_dict(), f'trained_models/{now}.pth')