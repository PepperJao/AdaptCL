import random
import numpy as np
    
from scipy.stats import norm
import matplotlib.pyplot as plt
import math
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random
from IPython.display import clear_output
import tensorflow
   
    
def plot_wb(model, fig_path, ranges=None):

    tmp = list(model.named_parameters())
    layers = []
    for i in range(0, len(tmp), 2):
        w, b = tmp[i], tmp[i + 1]
        if ("conv" in w[0] or "conv" in b[0]) or ("fc" in w[0] or "fc" in b[0]):
            layers.append((w, b))

    num_rows = len(layers)

    fig = plt.figure(figsize=(20, 40))

    i = 1
    for w, b in layers:
        w_flatten = w[1].flatten().detach().cpu().numpy()
        b_flatten = b[1].flatten().detach().cpu().numpy()

        fig.add_subplot(num_rows, 2, i)
        plt.title(w[0])
        plt.hist(w_flatten, bins=100, range=ranges);

        fig.add_subplot(num_rows, 2, i + 1)
        plt.title(b[0])
        plt.hist(b_flatten, bins=100, range=ranges);

        i += 2
    
    fig.tight_layout()
    plt.savefig(fig_path)
    plt.close()

def set_all_seed(seed_value=5):
    #have to be 5
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.random.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    tensorflow.random.set_seed(seed_value)
    
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def set_seed(seed=None, seed_torch=True):
    if seed is None:
        seed = np.random.choice(2 ** 32)
    random.seed(seed)
    np.random.seed(seed)
    if seed_torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    
    
    
def plot_mnist(data, nPlots=10):
    plt.figure(figsize=(12, 8))
    for ii in range(nPlots):
        plt.subplot(1, nPlots, ii + 1)
        plt.imshow(data[ii, 0], cmap="gray")
        plt.axis('off')
    plt.tight_layout
    plt.show()


def multi_task_barplot(accs, tasks, t=None):
    nTasks = len(accs)
    plt.bar(range(nTasks), accs, color='k')
    plt.ylabel('Testing Accuracy (%)', size=18)
    plt.xticks(range(nTasks), [f"{TN}\nTask {ii + 1}" for ii, TN in enumerate(tasks.keys())], size=18)
    plt.title(t)
    plt.show()


def plot_task(data, samples_num):
    plt.plot(figsize=(12, 6))
    for ii in range(samples_num):
        plt.subplot(1, samples_num, ii + 1)
        plt.imshow(data[ii][0], cmap="gray")
        plt.axis('off')
    plt.show()
    
    
def load_data(mnist_train, mnist_test, verbose=False, asnumpy=True):

    x_traint, t_traint = mnist_train.data, mnist_train.targets
    x_testt, t_testt = mnist_test.data, mnist_test.targets

    if asnumpy:
    # Fix dimensions and convert back to np array for code compatability
    # We aren't using torch dataloaders for ease of use
        x_traint = torch.unsqueeze(x_traint, 1)
        x_testt = torch.unsqueeze(x_testt, 1)
        x_train, x_test = x_traint.numpy().copy(), x_testt.numpy()
        t_train, t_test = t_traint.numpy().copy(), t_testt.numpy()
    else:
        x_train, t_train = x_traint, t_traint
        x_test, t_test = x_testt, t_testt

    if verbose:
        print(f"x_train dim: {x_train.shape} and type: {x_train.dtype}")
        print(f"t_train dim: {t_train.shape} and type: {t_train.dtype}")
        print(f"x_train dim: {x_test.shape} and type: {x_test.dtype}")
        print(f"t_train dim: {t_test.shape} and type: {t_test.dtype}")

    return x_train, t_train, x_test, t_test


def permute_mnist(mnist, seed, verbose=False):

    np.random.seed(seed)
    if verbose: print("starting permutation...")
    h = w = 28
    perm_inds = list(range(h*w))
    np.random.shuffle(perm_inds)
    # print(perm_inds)
    perm_mnist = []
    for set in mnist:
        num_img = set.shape[0]
        flat_set = set.reshape(num_img, w * h)
#         perm_mnist.append(flat_set[:, perm_inds].reshape(num_img, 1, w, h))
        perm_mnist.append(flat_set[:, perm_inds].reshape(num_img, 1, w, h))
    if verbose: print("done.")
    return perm_mnist




    print(f'Random seed {seed} has been set.')
    

# In case that `DataLoader` is used
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
def set_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("GPU is not enabled in this notebook. \n"
          "If you want to enable it, in the menu under `Runtime` -> \n"
          "`Hardware accelerator.` and select `GPU` from the dropdown menu")
    else:
        print("GPU is enabled in this notebook. \n"
          "If you want to disable it, in the menu under `Runtime` -> \n"
          "`Hardware accelerator.` and select `None` from the dropdown menu")

    return device