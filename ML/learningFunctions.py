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


def writeFeatherLog(content, path):
    infoFrame = pd.DataFrame(content)
    infoFrame.to_feather(path)


def writeLog(content, logPath, resetLog = False):
    # Open Log
    modelLog = open(logPath, 'a+')
    if resetLog:
        modelLog.truncate(0)

    modelLog.write(content + "\n")
    modelLog.close()

def calculateAccuracy(modelIn, modelOut, labels):
    max_scores, max_idx = modelOut.max(dim = 1)
    n = modelIn.size(0)
    assert( n == max_idx.size(0))
    # Calculate acc
    numberCorrect = (max_idx == labels).sum().item()
    numberPredict = n
    phaseAcc = numberCorrect / n
    #print("Phase acc: ", phaseAcc)
    return numberCorrect, numberPredict, phaseAcc

def saveModel(model, modelName, modelPath, loss, phase, epoch, modelLog = None):
    name = modelName + "_best_" + phase + "_" + str(epoch) + "_" + str(loss) + ".pth"
    savePath = os.path.join(modelPath, name)
    if modelLog is not None:
            writeLog(content  = "Saving " + phase + " model with loss: " + str(loss), logPath  = modelLog)
            writeLog(content  = "Saving model: " + savePath, logPath  = modelLog)

    else:
        tqdm.write("Saving " + phase + " model with loss: " + str(loss))
        tqdm.write("Saving model: " + savePath)

    torch.save(model.state_dict(), savePath)


def trainModel(model, optimizer, criterion, startEpoch, endEpoch, trainLoader, savePath, modelName, device, validLoader = None, testLoader = None):
    # Loss storage
    bestTrain = np.inf
    bestValid = np.inf
    # Make text file to store generic outputs
    modelLogPath = os.path.join(savePath, modelName + "_textLog.txt")
    featherPath = os.path.join(savePath, modelName + ".feather")

    writeLog(
        content  = "Training Model: " + modelName,
        logPath  = modelLogPath,
        resetLog = True,
    )
    modelInfo = {
        "trainLoss": [],
        "trainAcc": [],
        "trainTime": [],
        "validLoss": [],
        "validAcc": [],
        "validTime": [],
    }
    loaders = {
        'train': trainLoader,
        'valid': validLoader
    }
    # Train the model
    for i in tqdm(range(startEpoch, endEpoch), desc = "Epochs"):
        writeLog(
            content  = "Starting Epoch " + str(i + 1),
            logPath  = modelLogPath
        )
        for phase in ['train', 'valid']:
            phaseStart = time()
            dataLoader = loaders[phase]
            if phase == "train":
                model.train()
            else:
                model.eval()

            if phase == "valid":
                # Only do validation sometimes
                if i % 2 != 0:
                    #tqdm.write("Not doing validation")
                    writeLog(
                        content  = "Not doing validation",
                        logPath  = modelLogPath
                    )
                    dataLoader = None
                else:
                    #tqdm.write("Doing validation")
                    writeLog(
                        content  = "Doing validation",
                        logPath  = modelLogPath
                    )
            # Running loss is set to None so there will be some non-0 value if the validation loop does not run
            runningLoss = None
            if dataLoader is not None:
                runningLoss = 0
                correctPred = 0
                totalPred   = 0


                for step, (x, labels) in enumerate( tqdm(dataLoader, desc = phase + " loader", leave = False) ):
                    # TO DEVICE HERE
                    with torch.set_grad_enabled(phase == "train"):

                        x = x.to(device)
                        labels = labels.to(device)

                        optimizer.zero_grad()
                        predictions = model(x)
                        loss = criterion(predictions[0], labels)

                        numberCorrect, numberPredict, phaseAcc = calculateAccuracy(x.detach(), predictions[0].detach(), labels.detach())
                        correctPred += numberCorrect
                        totalPred   += numberPredict
                        totalLoss = loss.item() * x.shape[0]

                        # tqdm.write(phase + " LOSS: " + str(totalLoss))
                        runningLoss += totalLoss

                        if phase == "train":
                            loss.backward()
                            optimizer.step()
            # End if - we have looped through the data loader for this phase.
            phaseTime = time() - phaseStart
            phaseAcc = None
            if totalPred != 0:
                phaseAcc = correctPred / totalPred
            #tqdm.write( phase + " ended, loss: " + str(runningLoss) + " acc: " + str(phaseAcc))
            writeLog(
                content  = phase + " ended, loss: " + str(runningLoss) + " acc: " + str(phaseAcc),
                logPath  = modelLogPath
            )

            if phase == "train":
                modelInfo['trainAcc'].append(phaseAcc)
                modelInfo['trainLoss'].append(runningLoss)
                modelInfo['trainTime'].append(phaseTime)

                if runningLoss <= bestTrain:
                    writeLog(
                        content  = "New best train model: " + str(runningLoss) + " beats: " + str(bestTrain),
                        logPath  = modelLogPath
                    )

                    #tqdm.write("New best train model: " + str(runningLoss) + " beats: " + str(bestTrain))
                    bestTrain = runningLoss
                    saveModel(model, modelName, savePath, runningLoss, phase, i, modelLogPath)
            elif phase == "valid":
                if runningLoss is None:
                    modelInfo['validAcc'].append(None)
                    modelInfo['validLoss'].append(None)
                    modelInfo['validTime'].append(None)
                else:
                    # Valid has actually ran
                    modelInfo['validAcc'].append(phaseAcc)
                    modelInfo['validLoss'].append(runningLoss)
                    modelInfo['validTime'].append(phaseTime)

                    if runningLoss <= bestValid:
                        writeLog(
                            content  = "New best valid model: " + str(runningLoss) + " beats: " + str(bestValid),
                            logPath  = modelLogPath
                        )

                        #tqdm.write("New best valid model: " + str(runningLoss) + " beats: " + str(bestValid))
                        bestValid = runningLoss
                        saveModel(model, modelName, savePath, runningLoss, phase, i, modelLogPath)
        # Feather log should fire on each epoch right?
        writeFeatherLog(modelInfo, featherPath)
    # Epochs have finished.
    writeLog(
        content  = "Finished Training",
        logPath  = modelLogPath
    )


