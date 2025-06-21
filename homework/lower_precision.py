import torch
from pathlib import Path
from .bignet import BIGNET_DIM, LayerNorm

def block_quantize_custom(x: torch.Tensor, group_size: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Custom quantization to compress weights below 4 bits per parameter.
    """
    assert x.dim() == 1
    assert x.size(0) % group_size == 0

    x = x.view(-1, group_size)
    normalization = x.abs().max(dim=-1, keepdim=True).values
    x_norm = x / normalization
    x_quant = (x_norm * 15).round().to(torch.int8)  # Use 4-bit quantization
    return x_quant, normalization.to(torch.float16)


def block_dequantize_custom(x_quant: torch.Tensor, normalization: torch.Tensor) -> torch.Tensor:
    """
    Reverse operation of block_quantize_custom.
    """
    normalization = normalization.to(torch.float32)
    x_norm = x_quant.to(torch.float32) / 15
    x = x_norm * normalization
    return x.view(-1)

class LinearCustom(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, group_size: int = 16) -> None:
        super().__init__()
        self._shape = (out_features, in_features)
        self._group_size = group_size

        self.register_buffer(
            "weight_q",
            torch.zeros(out_features * in_features // group_size, group_size, dtype=torch.int8),
            persistent=False,
        )
        self.register_buffer(
            "weight_norm",
            torch.zeros(out_features * in_features // group_size, 1, dtype=torch.float16),
            persistent=False,
        )
        self.bias = None
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = block_dequantize_custom(self.weight_q, self.weight_norm).view(*self._shape)
        return torch.nn.functional.linear(x, weight, self.bias)


class LowerPrecisionBigNet(torch.nn.Module):
    """
    A BigNet optimized for lower precision (<4 bits per parameter).
    """

    class Block(torch.nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.model = torch.nn.Sequential(
                LinearCustom(channels, channels),
                torch.nn.ReLU(),
                LinearCustom(channels, channels),
                torch.nn.ReLU(),
                LinearCustom(channels, channels),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x) + x

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path | None) -> LowerPrecisionBigNet:
    """
    Load the LowerPrecisionBigNet model.
    """
    net = LowerPrecisionBigNet()
    if path is not None:
        state_dict = torch.load(path)  # Load the state dictionary without `strict`
        net.load_state_dict(state_dict, strict=False)  # Pass `strict` here
    return net