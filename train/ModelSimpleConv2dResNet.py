import torch
from einops import rearrange, repeat
from torch import nn as nn
from torchvision.models import resnet18
from torchvision.models.resnet import BasicBlock

""" SHARED ENCODER
Model that uses a simple Conv2d layer followed by a ResNet block.
"""


class ModelSimpleConv2dResNet(nn.Module):
    def __init__(self, grid_size: tuple[int, int], out_features: int) -> None:
        super().__init__()

        self.grid_size = grid_size

        self.addition_parameters = nn.Parameter(
            torch.randn(10, self.grid_size[0], self.grid_size[1])
        )

        resnet = resnet18()
        reslayer = resnet._make_layer(BasicBlock, 32, 10)
        reslayer = nn.Sequential(reslayer[1:])

        self.layer1 = nn.Sequential(
            nn.Conv2d(6 + 4 + 10, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            reslayer,
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
        y2 = self.layer2(rearrange(y1, "b c h w -> b (c h w)"))
        return y2
