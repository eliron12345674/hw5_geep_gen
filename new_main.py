from typing import List
import torch
from torch import Tensor, nn
from torch.nn import functional as F
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

DATASET_PATH = "../data"
savedModels_PATH = "finalProduct1"
lossGraph_PATH = "finalGraph"
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

def add_uniform_noise(sample):
    return sample.float() + torch.rand_like(sample.float())

def scale_to_unit_interval(sample):
    return sample / 255.0
cuda = True
DEVICE = torch.device("cuda" if cuda else "cpu")


batch_size = 128

x_dim  = 784
hidden_dim = 128
latent_dim = 64

lr = 1e-3

epochs = 10

static_var = 0.1

transform = transforms.Compose([
transforms.ToTensor(),
add_uniform_noise,
scale_to_unit_interval
])

train_dataset = MNIST(root=DATASET_PATH, train=True, transform=transform, download=True)
train_set, val_set = torch.utils.data.random_split(train_dataset, [50000, 10000])
test_set = MNIST(root=DATASET_PATH, train=False, transform=transform, download=True)

train_loader = data.DataLoader(train_set, batch_size=128, shuffle=True, drop_last=True, pin_memory=True)
val_loader = data.DataLoader(val_set, batch_size=128, shuffle=False, drop_last=False)
test_loader = data.DataLoader(test_set, batch_size=16, shuffle=False, drop_last=False)


class VAE(nn.Module):
    def __init__(self,
                 in_channels: int,
                 latent_dims: int,
                 hidden_dims: List[int] = None,
                 **kwargs) -> None:
        super(VAE, self).__init__()  # Initialize the parent class first

        self.latent_dim = latent_dims

        modules = []
        if hidden_dims is None:
            hidden_dims = [128, 256]

        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dims)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dims)

        # Build Decoder
        modules = []
        self.decoder_input = nn.Linear(latent_dims, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3,
                      kernel_size=3, padding=1),
            nn.Tanh())

    def encode(self, input: Tensor) -> List[Tensor]:
        batch_size = input.size(0)
        
        # Reshape input
        input = input.view(batch_size, 1, 28, 28)
        input = input.to(DEVICE)
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        z = self.reparameterize(mu, log_var)
        return mu, log_var, z

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)

        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)

        return z

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var, z = self.encode(input)
        return self.decode(z), mu, log_var

    def sample(self,
               num_samples: int,
               current_device: int, **kwargs) -> Tensor:

        z = torch.randn(num_samples,
                        self.latent_dim)
        z = z.to(current_device)
        samples = self.decode(z)

        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward(x)[0]
    

def plotLoss(list1, list2):
    """
    Plots two lists in relation to their indices with two y-axes.

    Parameters:
    list1 (list): The first list of data
    list2 (list): The second list of data
    """
    if len(list1) != len(list2):
        raise ValueError("Both lists must have the same length.")
    
    indices = range(len(list1))
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    color = 'tab:blue'
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('train loss', color=color)
    ax1.plot(indices, list1, marker='o', linestyle='-', color=color, label='List 1')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('test loss', color=color)
    ax2.plot(indices, list2, marker='s', linestyle='--', color=color, label='List 2')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Loss over Epochs')
    
    fig.tight_layout()
    plt.savefig(lossGraph_PATH)



from torch.optim import Adam
model = VAE(1,64)
BCE_loss = nn.BCELoss()

def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD


optimizer = Adam(model.parameters(), lr=lr)

print("Start training VAE...")
model.train()

def validate():
    overall_loss = 0
    model.eval()
    for batch_idx, (x, _) in enumerate(val_loader):
            x = x.view(batch_size, x_dim)
            x = x.to(DEVICE)

            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)
            
            overall_loss += loss.item()
    
    return overall_loss / len(val_loader)

def test():
    model.eval()
    overall_loss = 0
    for batch_idx, (x, _) in enumerate(test_loader):
            x = x.view(16, x_dim)
            x = x.to(DEVICE)

            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)
            
            overall_loss += loss.item()
    
    return overall_loss / len(test_loader)

def train():
    avgTrainLossList = []
    avgValLossList = []
    avgTestLossList = []
    for epoch in range(epochs):
        model.to(DEVICE)
        model.train()
        overall_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):

            x = x.view(batch_size, x_dim)
            x = x.to(DEVICE)

            optimizer.zero_grad()

            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)
            
            overall_loss += loss.item()
            
            loss.backward()
            optimizer.step()


        avgLoss = overall_loss / (batch_idx*batch_size)            
        avgTrainLossList.append(avgLoss)
        print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", avgLoss)

        TestLoss = test()
        avgTestLossList.append(TestLoss)

        plotLoss(avgTrainLossList, avgTestLossList)

        torch.save(model.state_dict(), savedModels_PATH)

    print("Finish!!")


train()