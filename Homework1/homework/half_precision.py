from pathlib import Path

import torch

from .bignet import BIGNET_DIM, LayerNorm  # noqa: F401


class HalfLinear(torch.nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        """
        Implement a half-precision Linear Layer.
        Feel free to use the torch.nn.Linear class as a parent class (it makes load_state_dict easier, names match).
        Feel free to set self.requires_grad_ to False, we will not backpropagate through this layer.
        """
        super().__init__(in_features,out_features,bias)
        self.weight.data = self.weight.data.to(torch.float16)
        if bias is True:
            self.bias.data = self.bias.data.to(torch.float16)
        self.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Hint: Use the .to method to cast a tensor to a different dtype (i.e. torch.float16 or x.dtype)
        # The input and output should be of x.dtype = torch.float32
        x_fp16 = x.to(torch.float16)
        out_fp16 = torch.nn.functional.linear(x_fp16,self.weight,self.bias)
        return out_fp16.to(torch.float32)


class HalfBigNet(torch.nn.Module):
    """
    A BigNet where all weights are in half precision. Make sure that the normalization uses full
    precision though to avoid numerical instability.
    """

    class Block(torch.nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.model = torch.nn.Sequential(
                HalfLinear(channels, channels),
                torch.nn.ReLU(),
                HalfLinear(channels, channels),
                torch.nn.ReLU(),
                HalfLinear(channels, channels),
            )

        def forward(self, x: torch.Tensor):
            return self.model(x) + x

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            self.Block(channels=BIGNET_DIM),
            LayerNorm(num_channels=BIGNET_DIM),
            self.Block(channels=BIGNET_DIM),
            LayerNorm(num_channels=BIGNET_DIM),
            self.Block(channels=BIGNET_DIM),
            LayerNorm(num_channels=BIGNET_DIM),
            self.Block(channels=BIGNET_DIM),
            LayerNorm(num_channels=BIGNET_DIM),
            self.Block(channels=BIGNET_DIM),
            LayerNorm(num_channels=BIGNET_DIM),
            self.Block(channels=BIGNET_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path | None) -> HalfBigNet:
    # You should not need to change anything here
    # PyTorch can load float32 states into float16 models
    net = HalfBigNet()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True))
    return net