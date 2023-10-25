from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import shutil
import random
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Dataset
from torchsummary import summary
from torchvision import transforms, datasets
import tensorflow
import pandas as pd
import os
from skimage import io, transform
from utils import *
set_all_seed(5)
from torch.utils.data import DataLoader, Subset


class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):

        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        img_name = img_name+'.png'
        image = io.imread(img_name)
        if self.transform:
            image = self.transform(image)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('long')
        landmarks = landmarks.squeeze()
        sample = [image, landmarks]
        return sample
    
def load_mnist_class(classes_to_load=[0, 1, 2, 3], BATCH_SIZE=64):
    set_all_seed(5)
    transform = transforms.Compose([transforms.Resize((32, 32)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])

    # Load the full MNIST dataset
    full_train_dataset = datasets.MNIST('./data', train=True, download=False, transform=transform)
    full_val_dataset = datasets.MNIST('./data', train=False, download=False, transform=transform)

    # Filter the dataset to include only the specified classes
    train_dataset = torch.utils.data.Subset(full_train_dataset, [i for i in range(len(full_train_dataset)) if full_train_dataset[i][1] in classes_to_load])
    val_dataset = torch.utils.data.Subset(full_val_dataset, [i for i in range(len(full_val_dataset)) if full_val_dataset[i][1] in classes_to_load])

    f, axarr = plt.subplots(2,2)
    axarr[0,0].imshow(train_dataset[0][0].squeeze(), cmap="gray")
    axarr[0,1].imshow(train_dataset[1][0].squeeze(), cmap="gray")
    axarr[1,0].imshow(train_dataset[2][0].squeeze(), cmap="gray")
    axarr[1,1].imshow(train_dataset[3][0].squeeze(), cmap="gray")
    np.vectorize(lambda ax:ax.axis('off'))(axarr);
    plt.savefig('data1.png')
    
    print("Image Shape: {}".format(train_dataset[0][0].size()))
    print("Training Set:   {} samples".format(len(train_dataset)))
    print("Validation Set:   {} samples".format(len(val_dataset)), end = '\n\n')
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return (train_loader, val_loader)

def load_permute_mnist_class(classes_to_load=[0, 1, 2, 3], BATCH_SIZE=64):
    set_all_seed(5)
    idx_permute = list(range(32*32))
    np.random.shuffle(idx_permute)
    perm_mnist = []
    transform = transforms.Compose([transforms.Resize((32,32)),
                                    transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x.view(-1)[idx_permute].view(1,32, 32)),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])

    # Load the full MNIST dataset
    full_train_dataset = datasets.MNIST('./data', train=True, download=False, transform=transform)
    full_val_dataset = datasets.MNIST('./data', train=False, download=False, transform=transform)

    # Filter the dataset to include only the specified classes
    train_dataset = torch.utils.data.Subset(full_train_dataset, [i for i in range(len(full_train_dataset)) if full_train_dataset[i][1] in classes_to_load])
    val_dataset = torch.utils.data.Subset(full_val_dataset, [i for i in range(len(full_val_dataset)) if full_val_dataset[i][1] in classes_to_load])

    f, axarr = plt.subplots(2,2)
    axarr[0,0].imshow(train_dataset[0][0].squeeze(), cmap="gray")
    axarr[0,1].imshow(train_dataset[1][0].squeeze(), cmap="gray")
    axarr[1,0].imshow(train_dataset[2][0].squeeze(), cmap="gray")
    axarr[1,1].imshow(train_dataset[3][0].squeeze(), cmap="gray")
    np.vectorize(lambda ax:ax.axis('off'))(axarr);
    plt.savefig('data1.png')
    
    print("Image Shape: {}".format(train_dataset[0][0].size()))
    print("Training Set:   {} samples".format(len(train_dataset)))
    print("Validation Set:   {} samples".format(len(val_dataset)), end = '\n\n')
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return (train_loader, val_loader)

def load_permute1_mnist_class(classes_to_load=[0, 1, 2, 3], BATCH_SIZE=64):
    set_all_seed(1)
    idx_permute = list(range(32*32))
    np.random.shuffle(idx_permute)
    perm_mnist = []
    transform = transforms.Compose([transforms.Resize((32,32)),
                                    transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x.view(-1)[idx_permute].view(1,32, 32)),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])

    # Load the full MNIST dataset
    full_train_dataset = datasets.MNIST('./data', train=True, download=False, transform=transform)
    full_val_dataset = datasets.MNIST('./data', train=False, download=False, transform=transform)

    # Filter the dataset to include only the specified classes
    train_dataset = torch.utils.data.Subset(full_train_dataset, [i for i in range(len(full_train_dataset)) if full_train_dataset[i][1] in classes_to_load])
    val_dataset = torch.utils.data.Subset(full_val_dataset, [i for i in range(len(full_val_dataset)) if full_val_dataset[i][1] in classes_to_load])

    f, axarr = plt.subplots(2,2)
    axarr[0,0].imshow(train_dataset[0][0].squeeze(), cmap="gray")
    axarr[0,1].imshow(train_dataset[1][0].squeeze(), cmap="gray")
    axarr[1,0].imshow(train_dataset[2][0].squeeze(), cmap="gray")
    axarr[1,1].imshow(train_dataset[3][0].squeeze(), cmap="gray")
    np.vectorize(lambda ax:ax.axis('off'))(axarr);
    plt.savefig('data1.png')
    
    print("Image Shape: {}".format(train_dataset[0][0].size()))
    print("Training Set:   {} samples".format(len(train_dataset)))
    print("Validation Set:   {} samples".format(len(val_dataset)), end = '\n\n')
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return (train_loader, val_loader)

def load_mnist(BATCH_SIZE=64):
    set_all_seed(5)
    transform = transforms.Compose([transforms.Resize((32,32)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])

    train_dataset = datasets.MNIST('./data', train=True, download=False, transform=transform)
    val_dataset = datasets.MNIST('./data', train=False, download=False, transform=transform)

    f, axarr = plt.subplots(2,2)
    axarr[0,0].imshow(train_dataset[0][0].squeeze(), cmap="gray")
    axarr[0,1].imshow(train_dataset[1][0].squeeze(), cmap="gray")
    axarr[1,0].imshow(train_dataset[2][0].squeeze(), cmap="gray")
    axarr[1,1].imshow(train_dataset[3][0].squeeze(), cmap="gray")
    np.vectorize(lambda ax:ax.axis('off'))(axarr);
    plt.savefig('data1.png')
    
    print("Image Shape: {}".format(train_dataset[0][0].size()))
    print("Training Set:   {} samples".format(len(train_dataset)))
    print("Validation Set:   {} samples".format(len(val_dataset)), end = '\n\n')
    
    # Create iterator.
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return (train_loader, val_loader)


def load_permute_mnist(BATCH_SIZE=64):
    set_all_seed(5)
    idx_permute = list(range(32*32))
    np.random.shuffle(idx_permute)
    perm_mnist = []
    transform = transforms.Compose([transforms.Resize((32,32)),
                                    transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x.view(-1)[idx_permute].view(1,32, 32)),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])


    train_dataset = datasets.MNIST('./data', train=True, download=False, transform=transform)
    val_dataset = datasets.MNIST('./data', train=False, download=False, transform=transform)
    
    f, axarr = plt.subplots(2,2)
    axarr[0,0].imshow(train_dataset[0][0].squeeze(), cmap="gray")
    axarr[0,1].imshow(train_dataset[1][0].squeeze(), cmap="gray")
    axarr[1,0].imshow(train_dataset[2][0].squeeze(), cmap="gray")
    axarr[1,1].imshow(train_dataset[3][0].squeeze(), cmap="gray")
    np.vectorize(lambda ax:ax.axis('off'))(axarr);
    plt.savefig('data2.png')

    print("Image Shape: {}".format(train_dataset[0][0].size()))
    print("Training Set:   {} samples".format(len(train_dataset)))
    print("Validation Set:   {} samples".format(len(val_dataset)), end = '\n\n')

    # Create iterator.
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return (train_loader, val_loader)

def load_rotated_mnist(BATCH_SIZE=64):
    set_all_seed(5)
    transform = transforms.Compose([transforms.Resize((32,32)),
                                    transforms.RandomRotation(degrees=30), 
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])
    

    train_dataset = datasets.MNIST('./data', train=True, download=False, transform=transform)
    val_dataset = datasets.MNIST('./data', train=False, download=False, transform=transform)

    f, axarr = plt.subplots(2,2)
    axarr[0,0].imshow(train_dataset[0][0].squeeze(), cmap="gray")
    axarr[0,1].imshow(train_dataset[1][0].squeeze(), cmap="gray")
    axarr[1,0].imshow(train_dataset[2][0].squeeze(), cmap="gray")
    axarr[1,1].imshow(train_dataset[3][0].squeeze(), cmap="gray")
    np.vectorize(lambda ax:ax.axis('off'))(axarr);
            
    print("Image Shape: {}".format(train_dataset[0][0].size()))
    print("Training Set:   {} samples".format(len(train_dataset)))
    print("Validation Set:   {} samples".format(len(val_dataset)), end = '\n\n')

    # Create iterator.
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return (train_loader, val_loader)

def load_rotated45_mnist(BATCH_SIZE=64):
    set_all_seed(5)
    transform = transforms.Compose([transforms.Resize((32,32)),
                                    transforms.RandomRotation(degrees=45), 
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])
    

    train_dataset = datasets.MNIST('./data', train=True, download=False, transform=transform)
    val_dataset = datasets.MNIST('./data', train=False, download=False, transform=transform)

    f, axarr = plt.subplots(2,2)
    axarr[0,0].imshow(train_dataset[0][0].squeeze(), cmap="gray")
    axarr[0,1].imshow(train_dataset[1][0].squeeze(), cmap="gray")
    axarr[1,0].imshow(train_dataset[2][0].squeeze(), cmap="gray")
    axarr[1,1].imshow(train_dataset[3][0].squeeze(), cmap="gray")
    np.vectorize(lambda ax:ax.axis('off'))(axarr);
            
    print("Image Shape: {}".format(train_dataset[0][0].size()))
    print("Training Set:   {} samples".format(len(train_dataset)))
    print("Validation Set:   {} samples".format(len(val_dataset)), end = '\n\n')

    # Create iterator.
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return (train_loader, val_loader)

def load_rotated90_mnist(BATCH_SIZE=64):
    set_all_seed(5)
    transform = transforms.Compose([transforms.Resize((32,32)),
                                    transforms.RandomRotation(degrees=90), 
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])
    

    train_dataset = datasets.MNIST('./data', train=True, download=False, transform=transform)
    val_dataset = datasets.MNIST('./data', train=False, download=False, transform=transform)

    f, axarr = plt.subplots(2,2)
    axarr[0,0].imshow(train_dataset[0][0].squeeze(), cmap="gray")
    axarr[0,1].imshow(train_dataset[1][0].squeeze(), cmap="gray")
    axarr[1,0].imshow(train_dataset[2][0].squeeze(), cmap="gray")
    axarr[1,1].imshow(train_dataset[3][0].squeeze(), cmap="gray")
    np.vectorize(lambda ax:ax.axis('off'))(axarr);
            
    print("Image Shape: {}".format(train_dataset[0][0].size()))
    print("Training Set:   {} samples".format(len(train_dataset)))
    print("Validation Set:   {} samples".format(len(val_dataset)), end = '\n\n')

    # Create iterator.
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return (train_loader, val_loader)

def load_inverted_mnist(BATCH_SIZE=64):
    set_all_seed(5)
    transform = transforms.Compose([transforms.Resize((32,32)),
                                    transforms.Lambda(lambda x: transforms.functional.invert(x)), 
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])
    

    train_dataset = datasets.MNIST('./data', train=True, download=False, transform=transform)
    val_dataset = datasets.MNIST('./data', train=False, download=False, transform=transform)

    f, axarr = plt.subplots(2,2)
    axarr[0,0].imshow(train_dataset[0][0].squeeze(), cmap="gray")
    axarr[0,1].imshow(train_dataset[1][0].squeeze(), cmap="gray")
    axarr[1,0].imshow(train_dataset[2][0].squeeze(), cmap="gray")
    axarr[1,1].imshow(train_dataset[3][0].squeeze(), cmap="gray")
    np.vectorize(lambda ax:ax.axis('off'))(axarr);
    plt.savefig('data3.png')
            
    print("Image Shape: {}".format(train_dataset[0][0].size()))
    print("Training Set:   {} samples".format(len(train_dataset)))
    print("Validation Set:   {} samples".format(len(val_dataset)), end = '\n\n')

    # Create iterator.
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return (train_loader, val_loader)


def load_domainnet_class(classes_to_load=["airplane", "basketball", "bear", "bee"], BATCH_SIZE=64, file_path="data/DomainNet/clipart"):
    set_all_seed(5)
    transform = transforms.Compose([transforms.Resize((32, 32)),
                                    transforms.ToTensor()])

    dataset = datasets.ImageFolder(root=file_path, transform=transform)

    # Filter the dataset to include only the specified classes
    selected_indices = [i for i in range(len(dataset)) if os.path.basename(os.path.dirname(dataset.imgs[i][0])) in classes_to_load]
    dataset = Subset(dataset, selected_indices)

    # Split the filtered dataset into training and validation sets
    num_samples = len(dataset)
    train_size = int(0.9 * num_samples)
    val_size = num_samples - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    f, axarr = plt.subplots(2, 2)
    axarr[0, 0].imshow(train_dataset[0][0].permute(1, 2, 0))
    axarr[0, 1].imshow(train_dataset[1][0].permute(1, 2, 0))
    axarr[1, 0].imshow(train_dataset[2][0].permute(1, 2, 0))
    axarr[1, 1].imshow(train_dataset[3][0].permute(1, 2, 0))
    np.vectorize(lambda ax: ax.axis('off'))(axarr)

    print("Image Shape: {}".format(train_dataset[0][0].size()))
    print("Training Set:   {} samples".format(len(train_dataset)))
    print("Validation Set:   {} samples".format(len(val_dataset), end='\n\n'))

    # Create iterators
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return (train_loader, val_loader)


def load_domain1(BATCH_SIZE=64, file_path = "data/DomainNet/domain1"):
    set_all_seed(5)
    transform = transforms.Compose([transforms.Resize((32,32)),
                                    transforms.ToTensor()])

    dataset = datasets.ImageFolder(root=file_path,transform=transform)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [4607, 400])

    f, axarr = plt.subplots(2,2)
    axarr[0,0].imshow(train_dataset[0][0].permute(1,2,0))
    axarr[0,1].imshow(train_dataset[1][0].permute(1,2,0))
    axarr[1,0].imshow(train_dataset[2][0].permute(1,2,0))
    axarr[1,1].imshow(train_dataset[3][0].permute(1,2,0))
    np.vectorize(lambda ax:ax.axis('off'))(axarr);

    print("Image Shape: {}".format(train_dataset[0][0].size()))
    print("Training Set:   {} samples".format(len(train_dataset)))
    print("Validation Set:   {} samples".format(len(val_dataset)), end = '\n\n')
    
    # Create iterator.
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return (train_loader, val_loader)

def load_domain2(BATCH_SIZE=64, file_path = "data/DomainNet/domain2"):
    set_all_seed(5)
    transform = transforms.Compose([transforms.Resize((32,32)),
                                    transforms.ToTensor()])

    dataset = datasets.ImageFolder(root=file_path,transform=transform)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [2100, 200])

    f, axarr = plt.subplots(2,2)
    axarr[0,0].imshow(train_dataset[0][0].permute(1,2,0))
    axarr[0,1].imshow(train_dataset[1][0].permute(1,2,0))
    axarr[1,0].imshow(train_dataset[2][0].permute(1,2,0))
    axarr[1,1].imshow(train_dataset[3][0].permute(1,2,0))
    np.vectorize(lambda ax:ax.axis('off'))(axarr);

    print("Image Shape: {}".format(train_dataset[0][0].size()))
    print("Training Set:   {} samples".format(len(train_dataset)))
    print("Validation Set:   {} samples".format(len(val_dataset)), end = '\n\n')
    
    # Create iterator.
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return (train_loader, val_loader)


def load_domain3(BATCH_SIZE=64, file_path = "data/DomainNet/domain3"):
    set_all_seed(5)
    transform = transforms.Compose([transforms.Resize((32,32)),
                                    transforms.ToTensor()])

    dataset = datasets.ImageFolder(root=file_path,transform=transform)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [1163, 100])

    f, axarr = plt.subplots(2,2)
    axarr[0,0].imshow(train_dataset[0][0].permute(1,2,0))
    axarr[0,1].imshow(train_dataset[1][0].permute(1,2,0))
    axarr[1,0].imshow(train_dataset[2][0].permute(1,2,0))
    axarr[1,1].imshow(train_dataset[3][0].permute(1,2,0))
    np.vectorize(lambda ax:ax.axis('off'))(axarr);

    print("Image Shape: {}".format(train_dataset[0][0].size()))
    print("Training Set:   {} samples".format(len(train_dataset)))
    print("Validation Set:   {} samples".format(len(val_dataset)), end = '\n\n')
    
    # Create iterator.
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return (train_loader, val_loader)

def load_apple(BATCH_SIZE=16, file_path = "data/Food/apple/apple.csv", folder_path = "data/Food/apple"):
    set_all_seed(5)
    transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize((32,32)),
                                    transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x[:3])])

    dataset = CustomDataset(file_path, folder_path, transform)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [55, 2])
    val_dataset = val_dataset + train_dataset

    f, axarr = plt.subplots(2,2)
    axarr[0,0].imshow(train_dataset[0][0].permute(1,2,0))
    axarr[0,1].imshow(train_dataset[1][0].permute(1,2,0))
    axarr[1,0].imshow(train_dataset[2][0].permute(1,2,0))
    axarr[1,1].imshow(train_dataset[3][0].permute(1,2,0))
    np.vectorize(lambda ax:ax.axis('off'))(axarr);

    print("Image Shape: {}".format(train_dataset[0][0].size()))
    print("Training Set:   {} samples".format(len(train_dataset)))
    print("Validation Set:   {} samples".format(len(val_dataset)), end = '\n\n')
    
    # Create iterator.
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return (train_loader, val_loader)

def load_bread(BATCH_SIZE=16, file_path = "data/Food/bread/bread.csv", folder_path = "data/Food/bread"):
    set_all_seed(5)
    transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize((32,32)),
                                    transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x[:3])])

    dataset = CustomDataset(file_path, folder_path, transform)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [91, 2])
    val_dataset = val_dataset + train_dataset

    f, axarr = plt.subplots(2,2)
    axarr[0,0].imshow(train_dataset[0][0].permute(1,2,0))
    axarr[0,1].imshow(train_dataset[1][0].permute(1,2,0))
    axarr[1,0].imshow(train_dataset[2][0].permute(1,2,0))
    axarr[1,1].imshow(train_dataset[3][0].permute(1,2,0))
    np.vectorize(lambda ax:ax.axis('off'))(axarr);

    print("Image Shape: {}".format(train_dataset[0][0].size()))
    print("Training Set:   {} samples".format(len(train_dataset)))
    print("Validation Set:   {} samples".format(len(val_dataset)), end = '\n\n')
    
    # Create iterator.
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return (train_loader, val_loader)


def load_apple2(BATCH_SIZE=16, file_path = "data/Food2/apple/apple.csv", folder_path = "data/Food2/apple"):
    set_all_seed(5)
    transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize((224,224)),
                                    transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x[:3])])

    dataset = CustomDataset(file_path, folder_path, transform)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [57+108+125, 0])
    val_dataset = val_dataset + train_dataset

    f, axarr = plt.subplots(2,2)
    axarr[0,0].imshow(train_dataset[0][0].permute(1,2,0))
    axarr[0,1].imshow(train_dataset[1][0].permute(1,2,0))
    axarr[1,0].imshow(train_dataset[2][0].permute(1,2,0))
    axarr[1,1].imshow(train_dataset[3][0].permute(1,2,0))
    np.vectorize(lambda ax:ax.axis('off'))(axarr);

    print("Image Shape: {}".format(train_dataset[0][0].size()))
    print("Training Set:   {} samples".format(len(train_dataset)))
    print("Validation Set:   {} samples".format(len(val_dataset)), end = '\n\n')
    
    # Create iterator.
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return (train_loader, val_loader)

def load_bread2(BATCH_SIZE=16, file_path = "data/Food2/bread/bread.csv", folder_path = "data/Food2/bread"):
    set_all_seed(5)
    transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize((224,224)),
                                    transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x[:3])])

    dataset = CustomDataset(file_path, folder_path, transform)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [93+69+69, 0])
    val_dataset = val_dataset + train_dataset

    f, axarr = plt.subplots(2,2)
    axarr[0,0].imshow(train_dataset[0][0].permute(1,2,0))
    axarr[0,1].imshow(train_dataset[1][0].permute(1,2,0))
    axarr[1,0].imshow(train_dataset[2][0].permute(1,2,0))
    axarr[1,1].imshow(train_dataset[3][0].permute(1,2,0))
    np.vectorize(lambda ax:ax.axis('off'))(axarr);

    print("Image Shape: {}".format(train_dataset[0][0].size()))
    print("Training Set:   {} samples".format(len(train_dataset)))
    print("Validation Set:   {} samples".format(len(val_dataset)), end = '\n\n')
    
    # Create iterator.
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return (train_loader, val_loader)