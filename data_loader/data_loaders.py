from base import BaseDataLoader
import numpy as np
import os
import torch
from torch.utils import data

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class CatsVsDogsDataLoader(BaseDataLoader):

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.data_dir = data_dir
        raw_data = np.load(os.path.join(data_dir, "processed_img_data.npy"), allow_pickle=True)
        X_data = torch.tensor([i[0] for i in raw_data]).view(-1, 1, 50, 50)
        X_data = X_data/255.0
        y_data = torch.tensor([np.argmax(i[1]) for i in raw_data], dtype=torch.long)

        self.dataset = data.TensorDataset(X_data, y_data)

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
