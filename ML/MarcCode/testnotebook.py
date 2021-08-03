import torch
from datasets.BavarianCrops_Dataset import BavarianCropsDataset
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
import scipy.signal as signal
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

DATA_ROOT = "S:/PhD Data/BavarianCrops"
CLASS_MAP = "S:/PhD Data/BavarianCrops/classmapping12.csv"
cropdata = BavarianCropsDataset(
    root = DATA_ROOT,
    region = "holl",
    partition = "train",
    classmapping = CLASS_MAP,
    samplet = None, 
    scheme = "random",
    mode = "traintest",
    seed = 10
)

numberClasses = cropdata.nclasses
print("Number Classes: ", numberClasses, "\n")


# For each class index, make a entry in crop strore
cropStore = {}
for i in range(cropdata.nclasses):
    cropStore[i] = []



# For each class index
for i in range(cropdata.nclasses):
    # Find locations in crop data where this is the index 
    classindex = np.where(cropdata.y == i + 1)[0]


    # Get data at those indices 
    for indx in classindex:
        x, y, id = cropdata[indx]
        # Save them to crop store
        cropStore[i].append(x)



# So at this point, each list in cropstore should be a list of temporal profiles for that class

loadedclasses = pd.read_csv(CLASS_MAP, index_col=0).sort_values(by="id")
print(loadedclasses)
print(cropStore[1][0].shape)

for i in range(0, 1):
    cropProfiles = cropStore[i]
    cropName     = loadedclasses.loc[loadedclasses['id'] == i]
    cropName     = cropName['classname'].values[0]
    
    smoothedNDVI = []
    rawNDVI      = []
    # For all profiles, work out NDVI
    for profiles in cropProfiles:
        b8   = profiles[:, 10].numpy()
        b4   = profiles[:,  6].numpy()
        ndvi = (b8 - b4) / (b8 + b4)
        ndviSeries = pd.Series(ndvi)
        #print(ndviSeries)
        
        # We could smooth? -- THIS BREAKS NOTEBOOK FOR SOME REASON, MAYBE BECAUSE OF PADDING?
        #smoothed = signal.savgol_filter(ndviSeries.values, 11, 3)
        #print(smoothed)
        #smoothedNDVI.append(smoothed)
        
        
        rawNDVI.append(ndviSeries)
    
    # Now average the pandas series?
    concatndvi = pd.concat(rawNDVI, axis = 1)
    # average
    averagendvi = concatndvi.mean( axis = 1)
    # print(averagendvi)
    
    # Average might be borked as time series gets longer? There are less frames contributing at each step.
    
    #ax.plot(concatFrames.mean( axis = 1), label = landCoverDefinitions[landClass], color = colour )
    #ax.set_title(landCoverDefinitions[landClass] )
    
    
    fig, axes = plt.subplots()
    axes.plot(averagendvi)
    
    print(averagendvi.values)
    
    smoothed = signal.savgol_filter(averagendvi.values, 11, 3)
    print(smoothed)

    axes.plot(smoothed)

plt.show()