import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        """Single MLP"""
        super().__init__()

        self.fc0 = nn.Linear(feature_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, feature_dim)

    def forward(self, inputs):
        act = F.gelu(self.fc0(inputs))
        return self.fc1(act)


class MixerLayer(nn.Module):
    def __init__(self, n_tokens, n_channels, token_mixer_dim, channel_mixer_dim):
        """Single Mixer layer

        Args:
            n_tokens (int): The number of tokens.
            n_channels (int): The number of channels of tokens, which means dimension of each token.
            token_mixer_dim (int): Hidden dimension of token mixer MLP.
            channel_mixer_dim (int): Hidden dimension of channel mixer MLP.
        """
        super().__init__()
        
        self.n_tokens = n_tokens
        self.n_channels = n_channels
        
        self.token_mixer_dim = token_mixer_dim
        self.channel_mixer_dim = channel_mixer_dim

        self.layer_norm0 = nn.LayerNorm(self.n_channels)
        self.layer_norm1 = nn.LayerNorm(self.n_channels)

        self.mlp0 = MLP(n_tokens, self.token_mixer_dim)
        self.mlp1 = MLP(n_channels, self.channel_mixer_dim)

    def forward(self, inputs):
        normed_inputs = self.layer_norm0(inputs)
        residual = normed_inputs.transpose(1, 2)
        residual = self.mlp0(residual)
        residual = residual.transpose(1, 2)
        inputs += residual
        
        normed_inputs = self.layer_norm1(inputs)
        residual = self.mlp1(normed_inputs)
        
        return inputs + residual


class Mixer(nn.Module):
    def __init__(self, image_size, patch_size, hidden_dim, n_classes, in_channels=3, n_blocks=16, token_mixer_dim=256, channel_mixer_dim=512):
        """Mixer model

        Args:
            image_size (int): The size of images.
            patch_size (int): The size of each patch. The size of image must be divisible by the size of patches
            hidden_dim (int): The number of channels of tokens, which means dimension of each token.
            n_classes (int): The number of output classes.
            in_channels (int, optional): The number of channels of input images. Defaults to 3.
            n_blocks (int, optional): The number of mixer layers. Defaults to 16.
            token_mixer_dim (int, optional): Hidden dimension of token mixer MLP.. Defaults to 256.
            channel_mixer_dim (int, optional): Hidden dimension of channel mixer MLP.. Defaults to 512.
        """
        super().__init__()

        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.n_patches = (image_size // patch_size) ** 2

        self.stem_conv = nn.Conv2d(
            in_channels,
            hidden_dim,
            kernel_size=(self.patch_size, self.patch_size),
            stride=(self.patch_size, self.patch_size),
        )
        self.blocks = nn.ModuleList([MixerLayer(self.n_patches, hidden_dim, token_mixer_dim, channel_mixer_dim) for _ in range(n_blocks)])
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        self.out_fc = nn.Linear(hidden_dim, n_classes)

    def forward(self, inputs):
        # get patches and change channel dim to last        
        patches = self.stem_conv(inputs)
        patches = patches.view(-1, self.hidden_dim, self.n_patches).transpose(1, 2)
        
        for layer in self.blocks:
            patches = layer(patches)
            
        normed_patches = self.layer_norm(patches)
        features = self.gap(normed_patches.transpose(1,2)).squeeze(-1)
        
        scores = self.out_fc(features)
        
        return scores, features
