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
savedModels_PATH = "finalProduct2"
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

cuda = True
DEVICE = torch.device("cuda" if cuda else "cpu")
print(DEVICE)

batch_size = 128

x_dim  = 784
hidden_dim = 128
latent_dim = 64

lr = 1e-3

epochs = 30

static_var = 0.1

transform = transforms.Compose([
transforms.ToTensor()
])

train_dataset = MNIST(root=DATASET_PATH, train=True, transform=transform, download=True)
train_set, val_set = torch.utils.data.random_split(train_dataset, [50000, 10000])
test_set = MNIST(root=DATASET_PATH, train=False, transform=transform, download=True)

train_loader = data.DataLoader(train_set, batch_size=128, shuffle=True, drop_last=True, pin_memory=True)
val_loader = data.DataLoader(val_set, batch_size=128, shuffle=False, drop_last=False)
test_loader = data.DataLoader(test_set, batch_size=16, shuffle=False, drop_last=False)

class Encoder(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, latent_dim, num_conv_layers):
        super(Encoder, self).__init__()
        self.num_conv_layers = num_conv_layers
        
        # Create a list to store convolutional layers
        self.convs = nn.ModuleList([nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1) for _ in range(num_conv_layers)])
        self.conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        
        # Adjust input_dim to account for the concatenated outputs
        self.FC_input = nn.Linear(input_dim // 4 * num_conv_layers, hidden_dim)
        self.FC_mean  = nn.Linear(hidden_dim, latent_dim)
        self.FC_var   = nn.Linear(hidden_dim, latent_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
        self.training = True
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Reshape input
        x = x.view(batch_size, 1, 28, 28)
        
        # Process input through each convolutional layer
        x = self.LeakyReLU(self.conv1(x))
        x = self.LeakyReLU(self.conv2(x))
        
        # Concatenate outputs along the channel dimension
        
        # Flatten the concatenated output
        x = x.view(batch_size, -1)
        
        # Fully connected layers
        x = self.LeakyReLU(self.FC_input(x))
        mean = self.FC_mean(x)
        log_var = self.FC_var(x)
                                                    
        return mean, log_var
    

class Decoder(nn.Module):
    
    def __init__(self, latent_dim, hidden_dim, output_dim, num_conv_layers):
        super(Decoder, self).__init__()
        
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        
        # Adjusting for the input size to the deconvolution layers
        deconv_input_dim = (output_dim // 4) * num_conv_layers
        self.FC_output = nn.Linear(hidden_dim, deconv_input_dim)
        
        
        self.deconvs = nn.ModuleList()
        for _ in range(num_conv_layers):
            self.deconvs.append(nn.ConvTranspose2d(num_conv_layers, 1, kernel_size=3, stride=2, padding=1, output_padding=1))
        self.deconv =  nn.ConvTranspose2d(num_conv_layers, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        # Final deconv layer to get to original input size
        self.deconv1 = nn.ConvTranspose2d(1, 1, kernel_size=3, stride=1, padding=1, output_padding=0)
        self.deconv2 = nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        x = self.LeakyReLU(self.FC_hidden(x))
        
        x = self.LeakyReLU(self.FC_output(x))
        
        # Reshape to the correct dimensions for deconvolution
        batch_size = x.size(0)
        x = x.view(batch_size, -1, 14, 14)  # Assuming the final conv layer outputs 7x7 feature maps
        
        # Process through each deconvolutional layer
        # conv_outputs = [(F.relu(conv(h))) for conv in self.deconvs]
        # x = torch.cat(conv_outputs, dim=1)
        x = self.LeakyReLU(self.deconv1(x))
        x = self.LeakyReLU(self.deconv2(x))
        # Flatten the concatenated output
        x = x.view(batch_size, -1)
    
        x = x.view(batch_size, 28, 28) 
        # Final deconvolution to match the original input dimensions
        x_hat = torch.sigmoid(x)
        x_hat = x_hat.view(batch_size, 784)
        return x_hat
    

class Model(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(Model, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
        
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(DEVICE)        # sampling epsilon        
        z = mean + var*epsilon                          # reparameterization trick
        return z
        
                
    def forward(self, x):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)
        x_hat = self.Decoder(z)
        
        return x_hat, mean, log_var
    

encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, num_conv_layers=1)
decoder = Decoder(latent_dim=latent_dim, hidden_dim = hidden_dim, output_dim = x_dim, num_conv_layers=1)

model = Model(Encoder=encoder, Decoder=decoder).to(DEVICE)

import matplotlib.pyplot as plt

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

# Example usage
list1 = [1, 4, 9, 16, 25]
list2 = [200, 300, 500, 700, 1100]
#plotLoss(list1, list2)


from torch.optim import Adam

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


#train()
import matplotlib.pyplot as plt
model.eval()

def show_image(x, idx):
    x = x.view(batch_size, 28, 28)

    fig = plt.figure()
    plt.imshow(x[idx].cpu().numpy())

def visualize_images(tensor):
    num_images, squared_size = tensor.shape
    img_size = int(squared_size**0.5)

    # Reshape each image to its 2D shape
    images = tensor.view(num_images, img_size, img_size)
    images = images.detach().numpy()
    # Plot all images
    fig, axes = plt.subplots(1, num_images, figsize=(num_images, 1))
    for i in range(num_images):
        ax = axes[i]
        ax.imshow(images[i], cmap='gray')
        ax.axis('off')

    plt.show()

with torch.no_grad():
    noise = torch.randn(batch_size, latent_dim).to(DEVICE)
    model.load_state_dict(torch.load(savedModels_PATH))
    generated_images = decoder(noise)



image1 = train_set[0]
image2 = train_set[90]

def transformsDigits(image1, image2):
    print(f"transforms from {image1[1]} to {image2[1]}")
    image1 = image1[0].to(DEVICE)
    image2 = image2[0].to(DEVICE)
    z1 = model.Encoder(image1)[0][0]
    z2 = model.Encoder(image2)[0][0]
    path = torch.stack([z1 + (i / (9)) * (z2 - z1) for i in range(10)])
    images = model.Decoder(path)
    visualize_images(images.cpu())


transformsDigits(image1, image2)

print('done')