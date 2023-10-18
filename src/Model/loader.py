import torch
from torch.utils.data import DataLoader, Dataset

import os

path = "data//processed_data/SMOTE//"


class DinoSet(Dataset):
    def __init__(self, path=path, mode='train'):
        '''
        mode should be 'train', 'val' or 'test' depending on what data we are loading
        '''
        super().__init__()
        self.path = path
        self.X_path = os.path.join(
            self.path, 'states', 'X_'+mode+'_tensor.pkl')
        self.y_path = os.path.join(
            self.path, 'actions', 'y_'+mode+'_tensor.pkl')
        self.X = torch.load(self.X_path)
        self.y = torch.load(self.y_path)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_batches(mode, batch_size=64, path=path):
    dino_dataset = DinoSet(path, mode=mode)
    loader = DataLoader(dino_dataset, batch_size=batch_size, shuffle=True)
    return loader
