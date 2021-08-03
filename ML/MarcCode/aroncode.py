""" Can we run this dataset code? """
import torch
from datasets.BavarianCrops_Dataset import BavarianCropsDataset
import matplotlib.pyplot as plt



    # if args.dataset == "BavarianCrops":
    #     root = os.path.join(args.dataroot,"BavarianCrops")

    #     #ImbalancedDatasetSampler
    #     test_dataset_list = list()
    #     for region in args.testregions:
    #         test_dataset_list.append(
    #             BavarianCropsDataset(root=root, region=region, partition=args.test_on,
    #                                         classmapping=args.classmapping, samplet=args.samplet,
    #                                  scheme=args.scheme,mode=args.mode, seed=args.seed)
    #         )

    #     train_dataset_list = list()
    #     for region in args.trainregions:
    #         train_dataset_list.append(
    #             BavarianCropsDataset(root=root, region=region, partition=args.train_on,
    #                                         classmapping=args.classmapping, samplet=args.samplet,
    #                                  scheme=args.scheme,mode=args.mode, seed=args.seed)
    #         )




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


cropset = torch.utils.data.DataLoader(dataset=cropdata, batch_size= 10)



X, Y, IDX = next(iter(cropset))

tempProfile = X[5]

print("CLASS: ", Y[5])

# print("TP: ", tempProfile.shape) # 147, 13 (13 bands)
# ['B1', 'B10', 'B11', 'B12', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9']


# NDVI is B8 - B4 / B8 + B4


b8 = tempProfile[:, 10].numpy()
b4 = tempProfile[:, 6].numpy()

print(b8.dtype)

NDVI = (b8 - b4) / (b8 + b4)
print(NDVI)

fig, ax = plt.subplots()

ax.plot(NDVI)
plt.show()






