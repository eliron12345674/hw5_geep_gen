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

def add_uniform_noise(sample):
    return sample.float() + torch.rand_like(sample.float())

def scale_to_unit_interval(sample):
    return sample / 255.0
cuda = True
DEVICE = torch.device("cuda" if cuda else "cpu")


batch_size = 128

x_dim  = 784
hidden_dim = 128
latent_dim = 128

lr = 1e-3

epochs = 30

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
test_loader = data.DataLoader(test_set, batch_size=128, shuffle=False, drop_last=False)

class Encoder(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, latent_dim, num_conv_layers):
        super(Encoder, self).__init__()
        self.num_conv_layers = num_conv_layers
        
        # Create a list to store convolutional layers
        self.convs = nn.ModuleList([nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1) for _ in range(num_conv_layers)])
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # Adjust input_dim to account for the concatenated outputs
        self.FC_input = nn.Linear(input_dim // 16 * num_conv_layers, hidden_dim)
        self.FC_mean  = nn.Linear(hidden_dim, latent_dim)
        self.FC_var   = nn.Linear(hidden_dim, latent_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
        self.training = True
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Reshape input
        x = x.view(batch_size, 1, 28, 28)
        
        # Process input through each convolutional layer
        conv_outputs = [self.pool(F.relu(conv(x))) for conv in self.convs]
        
        # Concatenate outputs along the channel dimension
        x = torch.cat(conv_outputs, dim=1)
        
        # Flatten the concatenated output
        x = x.view(batch_size, -1)
        
        # Fully connected layers
        h_ = self.LeakyReLU(self.FC_input(x))
        mean = self.FC_mean(h_)
        log_var = self.FC_var(h_)
                                                    
        return mean, log_var
    

class Decoder(nn.Module):
    
    def __init__(self, latent_dim, hidden_dim, output_dim, num_conv_layers):
        super(Decoder, self).__init__()
        
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        
        # Adjusting for the input size to the deconvolution layers
        deconv_input_dim = (output_dim // 16) * num_conv_layers
        self.FC_output = nn.Linear(hidden_dim, deconv_input_dim)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.deconvs = nn.ModuleList()
        for _ in range(num_conv_layers):
            self.deconvs.append(nn.ConvTranspose2d(num_conv_layers, 1, kernel_size=3, stride=2, padding=1, output_padding=1))
        self.deconv =  nn.ConvTranspose2d(num_conv_layers, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        # Final deconv layer to get to original input size
        self.final_deconv = nn.ConvTranspose2d(1, 1, kernel_size=3, stride=1, padding=1)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        h = self.LeakyReLU(self.FC_hidden(x))
        
        h = self.LeakyReLU(self.FC_output(h))
        
        # Reshape to the correct dimensions for deconvolution
        batch_size = h.size(0)
        h = h.view(batch_size, -1, 7, 7)  # Assuming the final conv layer outputs 7x7 feature maps
        
        # Process through each deconvolutional layer
        h = self.LeakyReLU(self.deconv(h))
        h = self.upsample(h)
        h = h.view(batch_size, 28, 28) 
        # Final deconvolution to match the original input dimensions
        x_hat = torch.sigmoid(h)
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
        x_hat            = self.Decoder(z)
        
        return x_hat, mean, log_var
    

encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, num_conv_layers=4)
decoder = Decoder(latent_dim=latent_dim, hidden_dim = hidden_dim, output_dim = x_dim, num_conv_layers=4)

model = Model(Encoder=encoder, Decoder=decoder).to(DEVICE)

from torch.optim import Adam

BCE_loss = nn.BCELoss()

def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD


optimizer = Adam(model.parameters(), lr=lr)

print("Start training VAE...")
model.train()

def train():
    for epoch in range(epochs):
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
            
        print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", overall_loss / (batch_idx*batch_size))
        
    print("Finish!!")
train()
import matplotlib.pyplot as plt
model.eval()

def show_image(x, idx):
    x = x.view(batch_size, 28, 28)

    fig = plt.figure()
    plt.imshow(x[idx].cpu().numpy())

with torch.no_grad():
    noise = torch.randn(batch_size, latent_dim).to(DEVICE)
    generated_images = decoder(noise)

show_image(generated_images, idx=12)


print('done')