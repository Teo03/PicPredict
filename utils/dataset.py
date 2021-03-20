import torch
from torch.utils.data import Dataset
from utils.data_utils import load_dataset

class DrawingsDataset(Dataset):
    def __init__(self, mtype):
        self.drawings, self.labels = load_dataset(mtype)
        
        self.drawings = torch.from_numpy(self.drawings)
        self.labels = torch.from_numpy(self.labels)

    def __getitem__(self, index):
        return self.drawings[index], self.labels[index]

    def __len__(self):
        return len(self.drawings)