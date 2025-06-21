import torch
from pathlib import Path
from .bignet import BIGNET_DIM, LayerNorm  # noqa: F401


class HalfLinear(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__(in_features, out_features, bias)
        # Convert weights and bias to half precision
        self.weight = torch.nn.Parameter(self.weight.to(torch.float16))
        if self.bias is not None:
            self.bias = torch.nn.Parameter(self.bias.to(torch.float16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input and output tensors are in float32
        x = x.to(torch.float32)
        weight = self.weight.to(torch.float32)
        bias = self.bias.to(torch.float32) if self.bias is not None else None
        return torch.nn.functional.linear(x, weight, bias)


class IdentityLayer(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class HalfBigNet(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, channels: int, use_identity: bool = False):
            super().__init__()
            if use_identity:
                self.model = torch.nn.Sequential(
                    IdentityLayer(),
                    IdentityLayer(),
                    IdentityLayer(),
                )
            else:
                self.model = torch.nn.Sequential(
                    HalfLinear(channels, channels),
                    torch.nn.ReLU(),
                    HalfLinear(channels, channels),
                    torch.nn.ReLU(),
                    HalfLinear(channels, channels),
                )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x) + x

    def __init__(self, identity_blocks: list[int] = None):
        super().__init__()
        identity_blocks = identity_blocks or []
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM, use_identity=0 in identity_blocks),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, use_identity=2 in identity_blocks),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, use_identity=4 in identity_blocks),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, use_identity=6 in identity_blocks),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, use_identity=8 in identity_blocks),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, use_identity=10 in identity_blocks),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path | None, identity_blocks: list[int] = None) -> HalfBigNet:
    net = HalfBigNet(identity_blocks=identity_blocks)
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True))
    return net