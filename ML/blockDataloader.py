"""
Our blocks are too large to combine and load for training - we need some kind of dataloader that can deal with our block structure

(Or we use a database?)

"""
import os
from pandas.core.indexes.api import get_objs_combined_axis
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm



class BlockDataset(Dataset):
    """ Dataset that can handle file split """

    def __init__(self, source, datasetType, blockInfo = None, generateIndex = False,  transform = None):
        """
        Args:
            source:        folder that contains the data
            datasetType:   determines the folder name
            transform:     optional transformation applied to samples
            blockInfo:     path to file that has index information (which index range in each block)
            generateIndex: Bool, if we should generate our own index and attach to block
        """
        self.source      = source
        self.datasetType = datasetType
        self.transform   = transform
        self.generateIndex = generateIndex
        # Given the data set type, we need to determine the number of samples we have and where they are

        self.maxLoaded    = 5
        self.loadedBlocks = []
        if blockInfo is None:
            self.blockInfo = None
            self.numSamples  = self.determineSamples()
        else:
            self.blockInfo = pd.read_feather(blockInfo)
            lastRow = self.blockInfo.tail(1)
            self.numSamples  = lastRow['endInd'].values[0] + 1

        # We shouldn't have to do this if block info is not null.


    def __len__(self):
        return self.numSamples


    def getBlockData(self, block, idx):
        # Passed a block data (dict) and index, get data at this index
        # print(block['block'].columns)
        blockData = block['block'].loc[idx]
        return blockData

    def loadNewBlock(self, blockName, idx):
        # Load blockName, get data at index and return it
        blockPath   = os.path.join(self.source, self.datasetType, blockName)

        #print("loading: ", blockPath)

        blockLoaded = pd.read_feather(blockPath)
        blockLoaded.set_index('loaderIndex', inplace = True)

        blockData = {'name' : blockName, 'block' : blockLoaded}

        self.loadedBlocks.append(blockData)

        if len(self.loadedBlocks) > self.maxLoaded:
            self.loadedBlocks.remove(self.loadedBlocks[0])

        return self.getBlockData(blockData, idx)

    def __getitem__(self, idx):
        # Given an index, we need to locate the correct block file and load that index from that file
        # Additionally, if a tuple is passed to this function, the returned data should be processed in a specific manner
        returnData = None
        if type(idx) is tuple:
            idx, full = idx
        else:
            full = False

        if self.blockInfo is not None:
            # We can resolve the id
            # Determine which block file the index is in
            dataRow = self.blockInfo.loc[ (idx <= self.blockInfo['endInd']) & (idx >= self.blockInfo['startInd']) ]
            #print(dataRow)
            if dataRow.empty:
                print(" Resolving ID gave empty frame: ", idx)
                print(self.blockInfo)
            else:
                # We found the block with the id
                blockName = dataRow.iloc[0]['blockName']
                #print("ID is in: ", blockName, idx)
                # We need to check if we have already loaded this block

                if len(self.loadedBlocks) == 0:
                    # There are no loaded blocks, we need to load this block
                    returnData = self.loadNewBlock(blockName, idx)
                else:
                    # There are some blocks in the cache
                    foundBlock = False
                    for loadedBlock in self.loadedBlocks:
                        if loadedBlock['name'] == blockName:
                            # The block we are looking for is loaded
                            foundBlock = True
                            returnData = self.getBlockData(loadedBlock, idx)
                    if foundBlock == False:
                        # The current block is not loaded we need to load it
                        returnData = self.loadNewBlock(blockName, idx)
        else:
            print("No block info, cannot resolve id")


        if returnData is not None:
            # Return data should be a single row from data frame
            # print("Return Data: ")
            # print(returnData)
            # Do additional formatting.

            if full is True:
                returnData = {
                    'class': returnData['cropClass'].astype(np.int16)
                }
            else:
                # We can calculate NDVI from band values and return that in a correctly formatted PyTorch tensor
                # ['date', 'B08', 'B04', 'B02', 'cropClass', 'foldNumber']

                sampleClass = returnData['cropClass'].astype(np.int32)

                b8 = returnData['B08'].astype(np.float32)
                b4 = returnData['B04'].astype(np.float32)
                b2 = returnData['B02'].astype(np.float32)

                # b8[b8 == 0] = np.nan
                # b4[b4 == 0] = np.nan
                # ndvi = (b8 - b4) / (b8 + b4)
                # ndvi = np.nan_to_num(ndvi)


                x = torch.from_numpy(np.array([b8, b4, b2]))


                # x = torch.unsqueeze(torch.from_numpy(ndvi), 0)
                #rint(x.shape)


                y = torch.tensor(sampleClass, dtype = torch.int64)



                return x, y

                # returnData = {
                #     'B02': returnData['B02'].astype(np.int16),
                #     'B04': returnData['B04'].astype(np.int16)
                # }

        else:
            print("Return data is none for some reason: ", idx)

        return returnData




    def determineSamples(self):
        # Data directory is determined through source + type, loop through all feather files and determine how many samples there are


        blockDirectory = os.path.join(self.source, self.datasetType)

        # Get list of block files
        blockList = os.listdir(blockDirectory)

        totalLength = 0
        blockInfo = []



        for ind, block in enumerate(tqdm(blockList, desc = " Loading Blocks")):

            blockPath = os.path.join(blockDirectory, block)
            loadBlock = pd.read_feather(blockPath)

            # print("Reading: ", blockPath)
            # print("Block Length: ", len(loadBlock))

            blockLength = len(loadBlock)
            customIndex = range(totalLength, totalLength + blockLength)

            if self.generateIndex:
                loadBlock['loaderIndex'] = customIndex
                loadBlock.to_feather(blockPath)



            if self.blockInfo is None:
                blockInfo.append({
                    'blockName' : block,
                    'startInd' : customIndex[0],
                    'endInd': customIndex[-1]
                })

            totalLength += len(loadBlock)
        # If we don't have input block info, generate the frame and save
        if self.blockInfo is None:
            self.blockInfo = pd.DataFrame(blockInfo)
            print(self.blockInfo)
            self.blockInfo.to_feather(os.path.join(self.source, self.datasetType + "Info.feather"))

        return totalLength


















