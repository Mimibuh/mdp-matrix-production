import torch
from einops import rearrange, repeat
from timm.models import VisionTransformer
from torch import nn as nn

"""SHARED ENCODER
Transformer-based encoder for grid observations with additional parameters.
"""


class ModelSimpleTransformer(nn.Module):
    def __init__(self, grid_size: tuple[int, int], out_features: int) -> None:
        super().__init__()

        self.grid_size = grid_size

        # learnable params
        self.addition_parameters = nn.Parameter(
            torch.randn(10, self.grid_size[0], self.grid_size[1])
        )
        trans = VisionTransformer(
            img_size=self.grid_size,
            patch_size=1,
            in_chans=32,
            embed_dim=100,
            depth=12,
            num_heads=4,
            mlp_ratio=4,
            num_classes=self.grid_size[0] * self.grid_size[1] * 32,
        )

        self.layer1 = nn.Sequential(
            nn.Conv2d(6 + 4 + 10, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            trans,
        )
        self.layer2 = nn.Sequential(
            nn.Linear(self.grid_size[0] * self.grid_size[1] * 32, out_features),
            nn.ReLU(),
        )

    def setup(self, action_space, observation_space, model_config):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.grid_size[0] * self.grid_size[1] * 6 + 4
        batch_size = x.shape[0]

        grid_obs = rearrange(
            x[:, : self.grid_size[0] * self.grid_size[1] * 6],
            "b (w h c) -> b c h w",
            h=self.grid_size[0],
            w=self.grid_size[1],
            c=6,
        )
        add1_ = repeat(
            x[:, self.grid_size[0] * self.grid_size[1] * 6 :],
            "b c -> b c h w",
            h=self.grid_size[0],
            w=self.grid_size[1],
            c=4,
        )
        input = torch.cat(
            (
                grid_obs,
                add1_,
                repeat(self.addition_parameters, "h w c -> b h w c", b=batch_size),
            ),
            dim=1,
        )
        y1 = self.layer1(input)
        # y1 = rearrange(y1, "b (c h w) -> b (c h w)")
        y2 = self.layer2(y1)
        return y2
