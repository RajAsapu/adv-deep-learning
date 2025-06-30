import abc

import torch
from torch import nn


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "AutoregressiveModel"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


class Autoregressive(abc.ABC):
    """
    Base class for all autoregressive models.
    Implement a specific model below.
    """

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Take a tensor x (B, h, w) if integers as input.
        Produce a probability over the next token as an output (B, h, w, n_token).
        Make sure the model is auto-regressive:
          - The first output result[:, 0, 0] does not depend on any input
          - The second output result[:, 0, 1] depends only on x[:, 0, 0]
          - etc.

        Hint 1: Flatten the tensor into a sequence.
        Hint 2: A positional embedding can help, but is not required.
        Hint 3: You need to shift the input sequence by 1 position. Do this after embedding the
                values, and before passing them through your model. (torch.concat or
                torch.nn.ConstantPad1d both work)
        """

    def generate(self, B: int = 1, h: int = 20, w: int = 30, device=None) -> torch.Tensor:  # noqa
        """
        Use your generative model to produce B new token images of size (B, h, w) and type (int/long).
        """


class AutoregressiveModel(nn.Module, Autoregressive):
    """
    Improved autoregressive model with positional embeddings, increased capacity,
    layer normalization, and dropout.
    """

    def __init__(self, d_latent: int = 256, n_tokens: int = 2 ** 10, n_layers: int = 12, n_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.n_tokens = n_tokens
        self.d_latent = d_latent

        # Initialize embedding with scaled initialization
        self.embedding = nn.Embedding(n_tokens, d_latent)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, 600, d_latent) * 0.02)  # Max seq_len = 20*30=600

        # Transformer with increased capacity
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_latent,
            nhead=n_heads,
            dim_feedforward=d_latent * 4,
            dropout=dropout,
            activation='gelu',  # Use GELU for better gradient flow
            batch_first=True,
            norm_first=True  # Pre-layer normalization
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output layer
        self.norm = nn.LayerNorm(d_latent)
        self.to_logits = nn.Linear(d_latent, n_tokens)

        # Initialize output layer
        nn.init.zeros_(self.to_logits.bias)
        nn.init.normal_(self.to_logits.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # x: (B, h, w) integer tokens
        B, h, w = x.shape
        seq_len = h * w
        x_flat = x.view(B, seq_len)  # (B, seq_len)

        # Embed tokens and add positional embeddings
        emb = self.embedding(x_flat)  # (B, seq_len, d_latent)
        emb = emb + self.pos_embedding[:, :seq_len, :]  # Add positional embeddings

        # Shift right for autoregressive prediction
        emb_shifted = torch.nn.functional.pad(emb, (0, 0, 1, 0), value=0)[:, :-1, :]

        # Causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()

        # Transformer forward pass
        out = self.transformer(emb_shifted, mask=mask)  # (B, seq_len, d_latent)
        out = self.norm(out)  # Apply final normalization
        logits = self.to_logits(out)  # (B, seq_len, n_tokens)
        logits = logits.view(B, h, w, self.n_tokens)

        return logits, {}

    def generate(self, B: int = 1, h: int = 20, w: int = 30, device=None) -> torch.Tensor:
        """
        Generate B new token images of size (B, h, w).
        """
        if device is None:
            device = next(self.parameters()).device
        self.eval()
        seq_len = h * w
        x_gen = torch.zeros((B, seq_len), dtype=torch.long, device=device)

        with torch.no_grad():
            for t in range(seq_len):
                # Get embeddings with positional information
                emb = self.embedding(x_gen[:, :t])  # (B, t, d_latent)
                emb = emb + self.pos_embedding[:, :t, :]  # Add positional embeddings

                # Pad for current step
                emb_padded = torch.nn.functional.pad(emb, (0, 0, 1, 0), value=0)[:, :t + 1, :]

                # Causal mask
                mask = torch.triu(torch.ones(t + 1, t + 1, device=device), diagonal=1).bool()

                # Forward pass
                out = self.transformer(emb_padded, mask=mask)
                out = self.norm(out)
                logits = self.to_logits(out[:, -1, :])  # (B, n_tokens)

                # Sample next token
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
                x_gen[:, t] = next_token

        return x_gen.view(B, h, w).to(torch.long)
