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
savedModels_PATH = "finalProduct"
trainGraph_PATH = "TrainGraph"
testGraph_PATH = "TestGraph"

def add_uniform_noise(sample):
    return sample.float() + torch.rand_like(sample.float())

def scale_to_unit_interval(sample):
    return sample / 255.0

def scale_to_256(sample):
    return sample * 255

cuda = True
DEVICE = torch.device("cuda" if cuda else "cpu")


batch_size = 128

x_dim  = 784
hidden_dim = 128
latent_dim = 128

lr = 1e-3

epochs = 40

static_var = 0.1

torch.manual_seed(torch.seed())

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
            self.deconvs.append(nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2, padding=0, output_padding=0))
        self.deconv =  nn.ConvTranspose2d(num_conv_layers, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        # Final deconv layer to get to original input size
        self.final_deconv = nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2, padding=0)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        h = self.LeakyReLU(self.FC_hidden(x))
        
        h = self.LeakyReLU(self.FC_output(h))
        
        # Reshape to the correct dimensions for deconvolution
        batch_size = h.size(0)
        h = h.view(batch_size, -1, 7, 7)  # Assuming the final conv layer outputs 7x7 feature maps
        
        # Process through each deconvolutional layer
        h = self.LeakyReLU(self.deconv(h))
                
        for inde, conv in enumerate(self.deconvs):
            h_hat = h[:,inde,:,:]
            #h_hat = h_hat.view(batch_size,14,14)
            if inde == 1-1:
                h_cova = conv(h_hat)
            else: 
                h_cova += conv(h_hat)

        h = h_cova
        #h = self.final_deconv(h)
        h = h[:,:,:28, :28]
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
        
        log_var = torch.full_like(mean, static_var)
        log_var = log_var.log()

        return x_hat, mean, log_var
    

encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, num_conv_layers=4)
decoder = Decoder(latent_dim=latent_dim, hidden_dim = hidden_dim, output_dim = x_dim, num_conv_layers=4)

model = Model(Encoder=encoder, Decoder=decoder).to(DEVICE)

import matplotlib.pyplot as plt

def plot_train_test(trainLoss, trainKLD, testLoss, testKLD):
    epochs = range(1, len(trainLoss) + 1)
    
    # Plot for Training data
    plt.figure(figsize=(12, 5))

    # Subplot 1: Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, trainLoss, label='Train Loss', color='blue', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()

    # Subplot 2: KLD
    plt.subplot(1, 2, 2)
    plt.plot(epochs, trainKLD, label='Train KLD', color='green', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('KLD')
    plt.title('Training KLD over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.savefig(trainGraph_PATH)
    plt.close()

    # Plot for Testing data
    plt.figure(figsize=(12, 5))

    # Subplot 1: Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, testLoss, label='Test Loss', color='orange', marker='x')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Testing Loss over Epochs')
    plt.legend()

    # Subplot 2: KLD
    plt.subplot(1, 2, 2)
    plt.plot(epochs, testKLD, label='Test KLD', color='red', marker='x')
    plt.xlabel('Epochs')
    plt.ylabel('KLD')
    plt.title('Testing KLD over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.savefig(testGraph_PATH)
    plt.close()

# Example usage
trainLoss = [0.9, 0.7, 0.6, 0.5]
trainKLD = [0.1, 0.2, 0.15, 0.1]
testLoss = [0.95, 0.75, 0.65, 0.55]
testKLD = [0.12, 0.22, 0.17, 0.12]

#plot_train_test(trainLoss, trainKLD, testLoss, testKLD)


from torch.optim import Adam

BCE_loss = nn.BCELoss()

def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss, KLD


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
            entropy, KLD = loss_function(x, x_hat, mean, log_var)
            loss = entropy + KLD
            
            overall_loss += loss.item()
    
    return overall_loss / len(val_loader)

def test():
    model.eval()
    overall_loss = 0
    overall_kld = 0
    for batch_idx, (x, _) in enumerate(test_loader):
            x = x.view(16, x_dim)
            x = x.to(DEVICE)

            x_hat, mean, log_var = model(x)
            entropy, KLD = loss_function(x, x_hat, mean, log_var)
            loss = entropy + KLD
            
            overall_loss += entropy.item()
            overall_kld += KLD.item()
    
    return overall_loss / len(test_loader), overall_kld / len(test_loader)

def train():
    avgTrainLossList = []
    avgTrainKLDList = []
    avgTestLossList = []
    avgTestKLDList = []

    for epoch in range(epochs):
        model.train()
        overall_loss = 0
        overall_kld = 0
        for batch_idx, (x, _) in enumerate(train_loader):

            x = x.view(batch_size, x_dim)
            x = x.to(DEVICE)

            optimizer.zero_grad()

            x_hat, mean, log_var = model(x)
            entropy, KLD = loss_function(x, x_hat, mean, log_var)
            loss = entropy + KLD
            
            overall_loss += entropy.item()
            overall_kld += KLD.item()

            
            loss.backward()
            optimizer.step()


        avgLoss = overall_loss / (batch_idx*batch_size)            
        avgTrainLossList.append(avgLoss)
        avgKLD = overall_kld / (batch_idx*batch_size)
        avgTrainKLDList.append(avgKLD)

        print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", avgLoss)

        TestLoss, TestKLD = test()
        avgTestLossList.append(TestLoss)
        avgTestKLDList.append(TestKLD)

        plot_train_test(avgTrainLossList, avgTrainKLDList, avgTestLossList, avgTestKLDList)

        torch.save(model.state_dict(), savedModels_PATH)

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
    model.load_state_dict(torch.load(savedModels_PATH))
    generated_images = decoder(noise)
    generated_images = scale_to_256(generated_images)

show_image(generated_images, idx=12)


print('done')