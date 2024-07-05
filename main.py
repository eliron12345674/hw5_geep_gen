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
SavedModels_PATH = "finalProduct"

def add_uniform_noise(sample):
    return sample.float() + torch.rand_like(sample.float())

def scale_to_unit_interval(sample):
    return sample / 255.0



def show_imgs(imgs):
    num_imgs = imgs.shape[0] if isinstance(imgs, torch.Tensor) else len(imgs)
    nrow = min(num_imgs, 4)
    ncol = int(math.ceil(num_imgs/nrow))
    imgs = torchvision.utils.make_grid(imgs, nrow=nrow, pad_value=128)
    imgs = imgs.clamp(min=0, max=255)
    np_imgs = imgs.cpu().numpy()
    plt.figure(figsize=(1.5*nrow, 1.5*ncol))
    plt.imshow(np.transpose(np_imgs, (1,2,0)), interpolation='nearest')
    plt.axis('off')
    plt.show()
    plt.close()


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super(Encoder, self).__init__()
        layers = []
        in_channels = input_dim[0]
        for h_dim in hidden_dims:
            layers.append(
                nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1)
            )
            layers.append(nn.ReLU())
            in_channels = h_dim
        
        self.conv_layers = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(2048, latent_dim)
        self.fc_logvar = nn.Linear(2048, latent_dim)

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, output_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, hidden_dims[-1] * (output_dim[1] // 4) * (output_dim[2] // 4))
        self.output_dim = output_dim
        
        layers = []
        hidden_dims.reverse()
        in_channels = hidden_dims[0]
        
        for i in range(1, len(hidden_dims)):
            layers.append(
                nn.ConvTranspose2d(in_channels, hidden_dims[i], kernel_size=3, stride=2, padding=1, output_padding=1)
            )
            layers.append(nn.ReLU())
            in_channels = hidden_dims[i]

        layers.append(
            nn.ConvTranspose2d(hidden_dims[-1], output_dim[0], kernel_size=3, stride=2, padding=1, output_padding=1)
        )
        layers.append(nn.Sigmoid())
        
        self.conv_layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), -1, self.output_dim[1] // 4, self.output_dim[2] // 4)
        x = self.conv_layers(x)
        x = x[:, :, :self.output_dim[1], :self.output_dim[2]]
        return x


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, recon_std=0.1):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dims, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dims, input_dim)
        self.recon_std = recon_std

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar

    def calc_elbo(self, x):
        recon_x, mu, logvar = self(x)
        recon_loss = F.mse_loss(recon_x, x, reduction='sum') / (2 * self.recon_std ** 2)
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        elbo = recon_loss + kld_loss
        return elbo, recon_loss, kld_loss

    def sample(self, num_samples, device):
        z = torch.randn(num_samples, latent_dim).to(device)
        samples = self.decoder(z)
        return samples


def train(model, dataloader, optimizer, device):
    model.train()
    train_elbo = 0
    for batch_idx, data in enumerate(dataloader):
        data = data[0].to(device)
        optimizer.zero_grad()
        elbo, recon_loss, kld_loss = model.calc_elbo(data)
        elbo.backward()
        train_elbo += elbo.item()
        optimizer.step()
    return train_elbo / len(dataloader.dataset)


def evaluate(model, dataloader, device):
    model.eval()
    val_elbo = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            data = data[0].to(device)
            elbo, recon_loss, kld_loss = model.calc_elbo(data)
            val_elbo += elbo.item()
    return val_elbo / len(dataloader.dataset)


if __name__ == "__main__":
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

    
    input_dim = (1, 28, 28)
    hidden_dims = [32, 64, 128]
    latent_dim = 20
    recon_std = 0.1
    learning_rate = 1e-3
    num_epochs = 100
    batch_size = 16

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(input_dim, hidden_dims, latent_dim, recon_std).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_elbos, val_elbos = [], []
    for epoch in range(num_epochs):
        train_elbo = train(model, train_loader, optimizer, device)
        val_elbo = evaluate(model, val_loader, device)
        train_elbos.append(train_elbo)
        val_elbos.append(val_elbo)
        print(f"Epoch {epoch + 1}, Train ELBO: {train_elbo:.4f}, Validation ELBO: {val_elbo:.4f}")

    # Plot ELBO curves
    plt.plot(train_elbos, label='Train ELBO')
    plt.plot(val_elbos, label='Validation ELBO')
    plt.legend()
    plt.show()

    # Generate samples
    samples = model.sample(1, device)
    show_imgs(samples.cpu())