"""
Train a TempCNN.
Needs to have ./ML subfolder with relevant code.

"""
import sys
sys.path.append("./ML/MarcCode")

from ML.MarcCode.models.TempCNN import TempCNN
from ML.blockDataloader import BlockDataset
from ML.learningFunctions import trainModel

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary

from tqdm.auto import tqdm
from time import time
import os




classLabels = {
    0: 'other',
    1: 'sugar beat',
    2: 'summer oat',
    3: 'meadow',
    4: 'rape',
    5: 'hop',
    6: 'winter spelt',
    7: 'winter triticale',
    8: 'beans',
    9: 'peas',
    10: 'potato',
    11: 'soybeans',
    12: 'asparagus',
    13: 'winter wheat',
    14: 'winter barley',
    15: 'winter rye',
    16: 'summer barley',
    17: 'maize'
}

# Fold source needs to be correct
FOLD_SOURCE = "/home/lunet/coab5/GEODATA/3 BAND"
SAVE_PATH = "/home/lunet/coab5/SAVED_MODELS"

#print(pd.read_feather("/home/lunet/coab5/GEODATA/Fold Blocks/TrainInfo.feather"))



if os.path.exists(FOLD_SOURCE):
    trainData = BlockDataset(
        source = FOLD_SOURCE,
        datasetType = "Train",
        #generateIndex = True,
        blockInfo =  os.path.join(FOLD_SOURCE, "TrainInfo.feather"),
    )
    validData = BlockDataset(
        source = FOLD_SOURCE,
        datasetType = "Valid",
        #generateIndex = True,
        blockInfo =  os.path.join(FOLD_SOURCE, "ValidInfo.feather"),
    )

    # trainIndices = range(4096)
    # validIndices = range(4096)

    # trainLoader = DataLoader(trainData, batch_size = 512, shuffle = False, sampler = torch.utils.data.SubsetRandomSampler(trainIndices))
    # validLoader = DataLoader(trainData, batch_size = 512, shuffle = False, sampler = torch.utils.data.SubsetRandomSampler(validIndices))
    trainLoader = DataLoader(trainData, batch_size = 2048, shuffle = False)
    validLoader = DataLoader(validData, batch_size = 2048, shuffle = False)




    modelargs = {
        'input_dim': 3,
        'nclasses': 18,
        'sequence_length': 127,
    }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = TempCNN(**modelargs)
    model.to(device)
    criterion   = nn.CrossEntropyLoss()
    optimizer   = optim.Adam(model.parameters())
    print("Training Model, Saving: ", SAVE_PATH)


    trainModel(
        model = model,
        optimizer = optimizer,
        criterion = criterion,
        startEpoch = 0,
        endEpoch = 45,
        trainLoader = trainLoader,
        savePath = SAVE_PATH,
        modelName = "TempCNN_3Channels_Fixed",
        validLoader = validLoader,
        device = device
    )
else:
    print("Something wrong with fold source, probably incorrect")











