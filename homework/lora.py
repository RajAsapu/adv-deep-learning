from pathlib import Path

import torch

from .bignet import BIGNET_DIM, LayerNorm  # noqa: F401
from .half_precision import HalfLinear

class LoRALinear(HalfLinear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_dim: int,
        bias: bool = True,
    ) -> None:
        super().__init__(in_features, out_features, bias)

        # Freeze base weights
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        # Initialize LoRA layers
        self.lora_a = torch.nn.Linear(in_features, lora_dim, bias=False)
        self.lora_b = torch.nn.Linear(lora_dim, out_features, bias=False)

        # Ensure LoRA layers are trainable and in float32 precision
        self.lora_a.weight = torch.nn.Parameter(self.lora_a.weight.to(torch.float32))
        self.lora_b.weight = torch.nn.Parameter(self.lora_b.weight.to(torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute LoRA output in float32
        lora_output = self.lora_b(self.lora_a(x.to(torch.float32)))

        # Compute base linear output in float16
        base_output = super().forward(x)

        # Combine outputs in float32 and cast back to input dtype
        combined_output = base_output + lora_output.to(base_output.dtype)
        return combined_output


class LoraBigNet(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, channels: int, lora_dim: int):
            super().__init__()
            self.model = torch.nn.Sequential(
                LoRALinear(channels, channels, lora_dim),
                torch.nn.ReLU(),
                LoRALinear(channels, channels, lora_dim),
                torch.nn.ReLU(),
                LoRALinear(channels, channels, lora_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x) + x

    def __init__(self, lora_dim: int = 32):
        super().__init__()
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM, lora_dim),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path | None) -> LoraBigNet:
    net = LoraBigNet()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True), strict=False)
    return net
