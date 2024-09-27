import torch
from torch import nn
import math
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):
    def __init__(self,):
        super().__init__(
            # (Batch_size, Channel, Height, Width) -> (Batch_size, 128, Height, Width)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            
            VAE_ResidualBlock(128, 128),
            
            VAE_ResidualBlock(128, 128),

            # (Batch_size, 128, Height, Width) -> (Batch_size, 128, Height/2, Width/2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            # (Batch_size, 128, Height/2, Width/2) -> (Batch_size, 256, Height/2, Width/2)
            VAE_ResidualBlock(128, 256),
            # (Batch_size, 256, Height/2, Width/2) -> (Batch_size, 256, Height/2, Width/2)
            VAE_ResidualBlock(256, 256),
            # (Batch_size, 256, Height/2, Width/2) -> (Batch_size, 256, Height/4, Width/4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            # (Batch_size, 256, Height/4, Width/4) -> (Batch_size, 512, Height/4, Width/4)
            VAE_ResidualBlock(256, 512),
            # (Batch_size, 512, Height/4, Width/4) -> (Batch_size, 512, Height/4, Width/4)
            VAE_ResidualBlock(512, 512),
            # (Batch_size, 512, Height/4, Width/4) -> (Batch_size, 512, Height/8, Width/8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
            # (Batch_size, 512, Height/8, Width/8) -> (Batch_size, 512, Height/8, Width/8)
            VAE_ResidualBlock(512, 512),
            # (Batch_size, 512, Height/8, Width/8) -> (Batch_size, 512, Height/8, Width/8)
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            # (Batch_size, 512, Height/8, Width/8) -> (Batch_size, 512, Height/8, Width/8)
            VAE_AttentionBlock(512),
            VAE_ResidualBlock(512, 512),
            # (Batch_size, 512, Height/8, Width/8) -> (Batch_size, 512, Height/8, Width/8)
            nn.GroupNorm(32, 512),

            nn.SiLU(),
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            nn.Conv2d(8, 8, kernel_size=1, padding=0),

        )
    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x: (Batch_size, Channel, Height, Width)
        # noise: (Batch_size, 8, Height/8, Width/8)
        for module in self:
            if getattr(module, 'stride', None) == (2, 2):
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)
        mean, log_variance = torch.chunk(x, 2, dim=1)

        log_variance = torch.clamp(log_variance, -30, 20)
        variance = log_variance.exp()
        std = variance.sqrt()
        x = mean + std * noise
        x *= 0.18215

        return x
