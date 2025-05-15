import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def linear_beta_schedule(timesteps):
    beta_start = 1e-4
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

class SimpleUnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(32),
            nn.Linear(32, 128),
            nn.ReLU()
        )

        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 1, 3, padding=1)

        self.time_dense = nn.Linear(128, 64)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        t_emb = self.time_dense(t_emb)[:, :, None, None]

        x = self.conv1(x) + t_emb
        x = F.relu(x)
        x = F.relu(self.conv2(x))
        return self.conv3(x)

class Diffusion:
    def __init__(self, timesteps=1000):
        self.timesteps = timesteps
        self.betas = linear_beta_schedule(timesteps).to(device)
        self.alphas = 1. - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, axis=0)# type:ignore

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha_cumprod = self.alpha_cumprod[t].sqrt().view(-1, 1, 1, 1)
        sqrt_one_minus = (1 - self.alpha_cumprod[t]).sqrt().view(-1, 1, 1, 1)
        return sqrt_alpha_cumprod * x0 + sqrt_one_minus * noise

    def p_sample(self, model, xt, t):
        betas_t = self.betas[t].view(-1, 1, 1, 1)
        sqrt_recip_alpha = (1. / self.alphas[t]).sqrt().view(-1, 1, 1, 1)
        sqrt_one_minus = (1 - self.alpha_cumprod[t]).sqrt().view(-1, 1, 1, 1)
        
        eps_theta = model(xt, t)
        mean = sqrt_recip_alpha * (xt - betas_t / sqrt_one_minus * eps_theta)

        noise = torch.randn_like(xt) if t[0] != 0 else torch.zeros_like(xt)
        return mean + betas_t.sqrt() * noise

def train(model, diffusion, dataloader, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        for x in tqdm(dataloader):
            x = x[0]  # images
            x = x.to(device)
            t = torch.randint(0, diffusion.timesteps, (x.size(0),), device=x.device).long()
            noise = torch.randn_like(x)
            x_noisy = diffusion.q_sample(x, t, noise)
            noise_pred = model(x_noisy, t)

            loss = F.mse_loss(noise_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}: loss = {loss.item():.4f}")

@torch.no_grad()
def sample(model, diffusion, shape):
    model.eval()
    img = torch.randn(shape).to(device)
    for t in reversed(range(diffusion.timesteps)):
        time = torch.full((shape[0],), t, device=device, dtype=torch.long)
        img = diffusion.p_sample(model, img, time)
    return img

def get_data():
    transform = transforms.Compose([
        transforms.ToTensor(),         # [0,1]
        transforms.Lambda(lambda x: x * 2 - 1)  # [-1,1] aralığına çek
    ])

    mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(mnist, batch_size=128, shuffle=True)
    return dataloader

model = SimpleUnet().to(device)
diffusion = Diffusion(timesteps=1000)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
dataloader = get_data()
train(model, diffusion, dataloader, optimizer, epochs=5)

@torch.no_grad()
def show_sample(model, diffusion, img_size=(1, 28, 28)):
    sampled_images = sample(model, diffusion, shape=(16, *img_size))
    sampled_images = (sampled_images + 1) * 0.5  # [0,1] aralığına getir

    grid_img = torch.cat([sampled_images[i] for i in range(16)], dim=2)
    plt.imshow(grid_img.permute(1, 2, 0).squeeze().cpu().numpy(), cmap='gray')
    plt.axis('off')
    plt.show()

show_sample(model, diffusion)