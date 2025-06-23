import torch
from einops import rearrange, repeat
from torch import nn as nn
from tools.global_config import custom_pretrained_models_path

""" SHARED ENCODER
Simple Conv2d model for grid-based environments:
- shallow model
- deeper model
- redecide model: redecision makes a logit decision, receives the state again concatenated with the logits and makes a new logit prediction

"""


class ModelSimpleConv2d(nn.Module):
    def __init__(self, grid_size: tuple[int, int], out_features: int) -> None:
        super().__init__()

        self.grid_size = grid_size

        self.addition_parameters = nn.Parameter(
            torch.rand(10, self.grid_size[0], self.grid_size[1])
        )

        self.layer1 = nn.Sequential(
            nn.Conv2d(6 + 4 + 10, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
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


class ModelSimpleConv2dDeeper(nn.Module):
    def __init__(self, grid_size: tuple[int, int], out_features: int) -> None:
        super().__init__()

        self.grid_size = grid_size

        self.addition_parameters = nn.Parameter(
            torch.rand(10, self.grid_size[0], self.grid_size[1])
        )

        self.layer1 = nn.Sequential(
            nn.Conv2d(6 + 4 + 10, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(self.grid_size[0] * self.grid_size[1] * 32, out_features),
            nn.ReLU(),
        )

    def setup(self, action_space, observation_space, model_config):
        if model_config["load_pretraining"]:
            path = (
                custom_pretrained_models_path
                / f"{model_config['pretrain_path']}_encoder.pth"
            )

            state_dict = torch.load(path)

            # Strip 'model.' prefix if present
            if any(k.startswith("model.") for k in state_dict):
                state_dict = {k[len("model.") :]: v for k, v in state_dict.items()}

            self.load_state_dict(state_dict)

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
                repeat(self.addition_parameters, "c h w -> b c h w", b=batch_size),
            ),
            dim=1,
        )
        y1 = self.layer1(input)
        y2 = self.layer2(rearrange(y1, "b c h w -> b (c h w)"))
        return y2


class ModelSimpleConv2dDeeperRedecide(nn.Module):
    def __init__(self, grid_size: tuple[int, int], out_features: int) -> None:
        super().__init__()

        self.grid_size = grid_size

        self.addition_parameters = nn.Parameter(
            torch.rand(10, self.grid_size[0], self.grid_size[1])
        )

        self.num_actions = self.grid_size[0] * self.grid_size[1] + 3
        # self.pos_actions_for_one_cell = (self.grid_size[0] * self.grid_size[1]) * 2 + 3

        self.layer1 = nn.Sequential(
            nn.Conv2d(
                6 + 4 + 10 + self.num_actions * 2,
                35,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(35, 34, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(34, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(self.grid_size[0] * self.grid_size[1] * 32, out_features),
            nn.ReLU(),
        )

    def setup(self, action_space, observation_space, model_config):
        if model_config["load_pretraining"]:
            path = (
                custom_pretrained_models_path
                / f"{model_config['pretrain_path']}_encoder.pth"
            )

            state_dict = torch.load(path)

            # Strip 'model.' prefix if present
            if any(k.startswith("model.") for k in state_dict):
                state_dict = {k[len("model.") :]: v for k, v in state_dict.items()}

            self.load_state_dict(state_dict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expected_features = (
            self.grid_size[0] * self.grid_size[1] * 6
            + 4
            + (self.grid_size[0] * self.grid_size[1] + 1) * self.num_actions
        )
        assert (
            x.shape[1] == expected_features
        ), f"Expected {expected_features} features, got {x.shape[1]}"

        batch_size = x.shape[0]

        grid_obs = rearrange(
            x[:, : self.grid_size[0] * self.grid_size[1] * 6],
            "b (w h c) -> b c h w",
            h=self.grid_size[0],
            w=self.grid_size[1],
            c=6,
        )
        add1_ = repeat(
            x[
                :,
                self.grid_size[0]
                * self.grid_size[1]
                * 6 : self.grid_size[0]
                * self.grid_size[1]
                * 6
                + 4,
            ],
            "b c -> b c h w",
            h=self.grid_size[0],
            w=self.grid_size[1],
            c=4,
        )
        redecide_ = rearrange(
            x[:, self.grid_size[0] * self.grid_size[1] * 6 + 4 : -self.num_actions],
            "b (w h c) -> b c h w",
            h=self.grid_size[0],
            w=self.grid_size[1],
            c=self.num_actions,
        )
        redecide_queue = repeat(
            x[:, -self.num_actions :],
            "b c -> b c h w",
            h=self.grid_size[0],
            w=self.grid_size[1],
            c=self.num_actions,
        )

        input = torch.cat(
            (
                grid_obs,
                add1_,
                repeat(self.addition_parameters, "h w c -> b h w c", b=batch_size),
                redecide_,
                redecide_queue,
            ),
            dim=1,
        )

        y1 = self.layer1(input)
        y2 = self.layer2(rearrange(y1, "b c h w -> b (c h w)"))

        return y2
