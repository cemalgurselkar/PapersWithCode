import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

img_size = 28
num_channel = 1

batch_size = 2
num_epoch = 10
lr = 0.001

num_timestep = 1000
beta_start = 0.0001
beta_end = 0.02
time_embed_dim = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class TimeEmbedding(nn.Module):
    """
    PE_(pos,2i) = sin(pos/10000^(2*i/d_model))
    PE_(pos,2i+1) = cos(pos/10000^(2*i/d_model))
    """
    def __init__(self, num_timestep, time_embed_dim):
        super().__init__()
        self.num_ts = num_timestep
        self.time_embedding = time_embed_dim

    def forward(self):
        time = torch.arange(self.num_ts, device=device).reshape(self.num_ts, 1)
        i = torch.arange(0, self.time_embedding, 2, device=device)
        denominator = torch.pow(10000, i/self.time_embedding)

        even_time_emb = torch.sin(time/denominator)
        odd_time_emb = torch.cos(time/denominator)

        stacked = torch.stack([even_time_emb, odd_time_emb], dim=2)
        time_embedding = torch.flatten(stacked, start_dim=1, end_dim=2)
        return time_embedding

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, bias=False, padding=1)

        self.bn0 = nn.BatchNorm2d(num_features=out_channels)
        self.time_embedding = TimeEmbedding(num_timestep=num_timestep, time_embed_dim=time_embed_dim)
        self.linear = nn.Linear(in_features=time_embed_dim, out_features=out_channels)

        self.conv1 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, bias=False, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        """
    block = conv0 -> batch0 -> relu
    block = block + time_embd
    blokc = conv1 -> batch1 -> relu
        """
        x = self.relu(self.bn0(self.conv0(x)))

        time_emb = self.time_embedding()[t]
        
        time_emb = self.linear(time_emb) # type: ignore
        time_emb = time_emb[:, :, None, None]

        x = x + time_emb
        
        x = self.relu(self.bn1(self.conv1(x)))
        return x

class DownSample(nn.Module): #Encoder
    def __init__(self, in_channel, out_channel):
        self.double_conv = DoubleConv(in_channels=in_channel, out_channels=out_channel)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x, t):
        convolved = self.double_conv(x, t)
        maxpooled = self.maxpool(convolved)
        return convolved, maxpooled

class UpSample(nn.Module): #Decoder
    def __init__(self, in_channel, out_channel):
        self.conv_transpose = nn.ConvTranspose2d(in_channels=in_channel, out_channels=out_channel, kernel_size=2, stride=2)
        self.double_conv = DoubleConv(in_channels=in_channel, out_channels=out_channel)
    
    def forward(self, x, t, connection):
        x = self.conv_transpose(x)
        x = torch.cat([x, connection], dim=1)
        x = self.double_conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, num_channel):
        super().__init__()
        self.num_channel = num_channel
        
        self.downsample0 = DownSample(in_channel=self.num_channel, out_channel=64)
        self.downsample1 = DownSample(in_channel=64, out_channel=128)

        self.bottleneck = DoubleConv(in_channels=128, out_channels=256)

        self.upsample0 = UpSample(in_channel=256, out_channel=128)
        self.upsample1 = UpSample(in_channel=128, out_channel=64)

        self.output = nn.Conv2d(in_channels=64, out_channels=num_channel, kernel_size=1)
    
    def forward(self, x, t):
        convolved0, maxpooled0 = self.downsample0(x,t)
        convolved1, maxpooled1 = self.downsample1(maxpooled0, t)

        x = self.bottleneck(maxpooled1, t)

        upsample0 = self.upsample0(x,t, convolved1)
        upsample1 = self.upsample1(upsample0, t, convolved0)

        x = self.output(upsample1)
        return x

unet_test = UNet(num_channel=num_channel).to(device=device)

def get_data(batch_size=16):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    loader = DataLoader(data, batch_size=batch_size, drop_last=True, shuffle=True)
    images, labels = next(iter(loader))
    return images, labels

images, labels = get_data(batch_size=batch_size)

class NoiseSchedular:
    def __init__(self):
        self.betas = torch.linspace(beta_start, beta_end, num_timestep)
        self.alphas = 1. - self.betas
        self.alphas_cum_prod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cum_prod = torch.sqrt(self.alphas_cum_prod)
        self.sqrt_one_minus_alphas_cum_prod = torch.sqrt(1. - self.alphas_cum_prod)
    
    def forward_diffusion(self, original, noise, t):
        """
        x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t)
        """
        sqrt_alphas_cum_prod_t = self.sqrt_alphas_cum_prod[t]
        sqrt_alphas_cum_prod_t = sqrt_alphas_cum_prod_t.to(device=device)

        sqrt_one_minus_alphas_cum_prod_t = self.sqrt_one_minus_alphas_cum_prod[t]
        sqrt_one_minus_alphas_cum_prod_t = sqrt_one_minus_alphas_cum_prod_t.to(device=device).view(-1,1,1,1)

        noisy_image = sqrt_alphas_cum_prod_t * original + sqrt_one_minus_alphas_cum_prod_t * noise
        return noisy_image
    
    def backward_diffusion(self, current_image, predicted_noise, t):  #(1)
        denoised_image = (current_image - (self.sqrt_one_minus_alphas_cum_prod[t] * predicted_noise)) / self.sqrt_alphas_cum_prod[t]  #(2)
        denoised_image = 2 * (denoised_image - denoised_image.min()) / (denoised_image.max() - denoised_image.min()) - 1  #(3)
        
        current_prediction = current_image - ((self.betas[t] * predicted_noise) / (self.sqrt_one_minus_alphas_cum_prod[t]))  #(4)
        current_prediction = current_prediction / torch.sqrt(self.alphas[t])  #(5)
        
        if t == 0:  #(6)
            return current_prediction, denoised_image
        
        else:
            variance = (1 - self.alphas_cum_prod[t-1]) / (1. - self.alphas_cum_prod[t])  #(7)
            variance = variance * self.betas[t]  #(8)
            sigma = variance ** 0.5
            z = torch.randn(current_image.shape).to(device=device)
            current_prediction = current_prediction + sigma*z
            
            return current_prediction, denoised_image

noise_schedular = NoiseSchedular()

image = images[0]
noise = torch.randn_like(image)

plt.imshow(image.squeeze(), cmap="gray")
plt.show()
plt.imshow(noise.squeeze(), cmap="gray")
plt.show()