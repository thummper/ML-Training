from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch.nn as nn
import torch

class LandCoverDataset(Dataset):
    def __init__(self, dataFrame, transform = None):
        self.dataFrame = dataFrame
        self.transform =  transform
        self.maxLength = 72

    def __len__(self):
        # Should return length of dataframe
        return len(self.dataFrame.index)

    def __getitem__(self, idx):
        # Return item from index
        row   = self.dataFrame.iloc[idx]
        ndviData = row['NDVI']
        paddingAmount = self.maxLength - ndviData.shape[0]

        if paddingAmount < 0:
            ndviData = ndviData[:paddingAmount]
        else:
             paddingAmount = (0, paddingAmount)
             ndviData = np.pad(ndviData, paddingAmount, constant_values = 0)








        data  = torch.from_numpy(ndviData.astype(np.float32).reshape(-1, 72))
        label = torch.tensor(row['landClass'], dtype = torch.int64)

        return data, label

