import abc

import torch

from .ae import PatchAutoEncoder


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "BSQPatchAutoEncoder"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


def diff_sign(x: torch.Tensor) -> torch.Tensor:
    """
    A differentiable sign function using the straight-through estimator.
    Returns -1 for negative values and 1 for non-negative values.
    """
    sign = 2 * (x >= 0).float() - 1
    return x + (sign - x).detach()


class Tokenizer(abc.ABC):
    """
    Base class for all tokenizers.
    Implement a specific tokenizer below.
    """

    @abc.abstractmethod
    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Tokenize an image tensor of shape (B, H, W, C) into
        an integer tensor of shape (B, h, w) where h * patch_size = H and w * patch_size = W
        """

    @abc.abstractmethod
    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode a tokenized image into an image tensor.
        """


class BSQ(torch.nn.Module):
    def __init__(self, codebook_bits: int, embedding_dim: int):
        super().__init__()
        self.codebook_bits = codebook_bits
        self.embedding_dim = embedding_dim
        self.down_projection = torch.nn.Linear(embedding_dim, codebook_bits)
        self.up_projection = torch.nn.Linear(codebook_bits, embedding_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implement the BSQ encoder:
        - A linear down-projection into codebook_bits dimensions
        - L2 normalization
        - differentiable sign
        """
        x_shape = x.shape
        assert x_shape[-1] == self.embedding_dim, \
            f"Expected last dim {self.embedding_dim}, got {x_shape[-1]}"
        x = self.down_projection(x)  # (B, h, w, codebook_bits)
        x = torch.nn.functional.normalize(x, p=2, dim=-1)
        x = diff_sign(x)
        return x  # (B, h, w, codebook_bits)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implement the BSQ decoder:
        - A linear up-projection into embedding_dim should suffice
        """
        # Reshape the input tensor to ensure compatibility with up_projection
        x_shape = x.shape
        assert x_shape[-1] == self.codebook_bits, \
            f"Expected last dim {self.codebook_bits}, got {x_shape[-1]}"
        x = self.up_projection(x)  # (B, h, w, embedding_dim)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run BQS and encode the input tensor x into a set of integer tokens
        """
        return self._code_to_index(self.encode(x))

    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode a set of integer tokens into an image.
        """
        return self.decode(self._index_to_code(x))

    def _code_to_index(self, x: torch.Tensor) -> torch.Tensor:
        x = (x >= 0).int()
        return (x * 2 ** torch.arange(x.size(-1)).to(x.device)).sum(dim=-1)

    def _index_to_code(self, x: torch.Tensor) -> torch.Tensor:
        return 2 * ((x[..., None] & (2 ** torch.arange(self._codebook_bits).to(x.device))) > 0).float() - 1


class BSQPatchAutoEncoder(PatchAutoEncoder, Tokenizer):
    """
    Combine your PatchAutoEncoder with BSQ to form a Tokenizer.

    Hint: The hyper-parameters below should work fine, no need to change them
          Changing the patch-size of codebook-size will complicate later parts of the assignment.
    """

    def __init__(self, patch_size: int = 5, latent_dim: int = 128, codebook_bits: int = 10,
                 input_shape: tuple[int, int, int, int] = (1, 28, 28, 3)):
        super().__init__(patch_size=patch_size, latent_dim=latent_dim)
        self.bsq = BSQ(codebook_bits=codebook_bits, embedding_dim=latent_dim)
        self.input_shape = input_shape

    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Tokenize an image tensor into integer tokens using BSQ.
        """
        features = super().encode(x)  # (B, h, w, latent_dim)
        return self.bsq.encode_index(features)  # (B, h, w)

    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode integer tokens back into an image tensor using BSQ.
        """
        decoded_features = self.bsq.decode_index(x)  # (B, h, w, latent_dim)
        return super().decode(decoded_features)  # (B, H, W, C)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode the input tensor using the auto-encoder and BSQ quantizer.
        """
        # Encode with PatchAutoEncoder, then quantize with BSQ
        features = super().encode(x)
        return self.bsq.encode(features)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode the quantized features back into the original space.
        Ensure the output shape matches the input shape.
        """
        # Decode with BSQ, then reconstruct with PatchAutoEncoder
        decoded = self.bsq.decode(x)
        return super().decode(decoded)


    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Return the reconstructed image and a dictionary of additional loss terms you would like to
        minimize (or even just visualize).
        """
        recon = self.decode(self.encode(x))
        tokens = self.encode_index(x).flatten().to(torch.long)
        cnt = torch.bincount(tokens, minlength=2 ** self.bsq.codebook_bits)
        stats = {
            "cb0": (cnt == 0).float().mean().detach(),
            "cb2": (cnt <= 2).float().mean().detach(),
        }
        return recon, stats
