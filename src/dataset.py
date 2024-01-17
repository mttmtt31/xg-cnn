import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.ndimage import gaussian_filter

class HeatmapDataset(Dataset):
    def __init__(self, data_path, labels_path, g_filter:float=0.75, augmentation=None):
        # load the numpy arrays
        data = np.load(data_path)
        labels = np.load(labels_path)
        # apply gaussian smoothing filter
        data = gaussian_filter(data, sigma = (0, 0, g_filter, g_filter))
        # turn them into tensors
        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).float()
        # check augmentation 
        self.augmentation = augmentation
        # find real length
        self.real_length = self.data.shape[0]

    def __len__(self):
        return self.real_length * (self.augmentation + 1)
    
    def __getitem__(self, index):
        # check if augmentation is on
        if index >= self.real_length:
            # locate tensor at the right index
            tensor = self.data[index - self.real_length]
            # perform augmentation
            tensor = tensor.flip(dims = [2])
            # locate label at the right index
            label = self.labels[index - self.real_length]
        else:
            # locate tensor 
            tensor = self.data[index]
            # locate label
            label = self.labels[index]

        return tensor, label
    