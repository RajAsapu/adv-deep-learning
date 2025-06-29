import abc

import torch


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


class AutoregressiveModel(torch.nn.Module, Autoregressive):
    """
    Implement an auto-regressive model.
    The input is a set of patch tokens (integers), the output is an image of probability.
    You need to implicitly shift your inputs by one position in the forward pass.
    Make sure n_tokens matches your BSQ dimension (2**codebook_bits_).

    Hint: You will need the torch.nn.Embedding function
    Hint: You can use torch.nn.TransformerEncoderLayer if you'd like
    Hint: You can complete this homework without using positional embeddings
    """

    def __init__(self, d_latent: int = 128, n_tokens: int = 2 ** 10):
        super().__init__()
        self.n_tokens = n_tokens
        self.d_latent = d_latent
        self.embedding = torch.nn.Embedding(n_tokens, d_latent)
        self.transformer = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_latent, nhead=8, batch_first=True),
            num_layers=8
        )
        self.to_logits = torch.nn.Linear(d_latent, n_tokens)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
          # x: (B, h, w) integer tokens
          B, h, w = x.shape
          seq_len = h * w
          x_flat = x.view(B, seq_len)  # (B, seq_len)
          emb = self.embedding(x_flat)  # (B, seq_len, d_latent)
          # Shift right for autoregressive prediction (after embedding)
          emb_shifted = torch.nn.functional.pad(emb, (0, 0, 1, 0), value=0)[:, :-1, :]
          # Causal mask for TransformerEncoder: shape (seq_len, seq_len)
          mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
          out = self.transformer(emb_shifted, mask=mask)  # (B, seq_len, d_latent)
          logits = self.to_logits(out)  # (B, seq_len, n_tokens)
          logits = logits.view(B, h, w, self.n_tokens)
          return logits, {}

    def generate(self, B: int = 1, h: int = 20, w: int = 30, device=None) -> torch.Tensor:
        """
        Use your generative model to produce B new token images of size (B, h, w) and type (int/long).
        """
        if device is None:
            device = next(self.parameters()).device
        self.eval()
        seq_len = h * w
        x_gen = torch.zeros((B, seq_len), dtype=torch.long, device=device)
        with torch.no_grad():
            for t in range(seq_len):
                # Shift right: only use generated tokens up to t
                x_shifted = torch.nn.functional.pad(x_gen, (1, 0), value=0)[:, :-1]
                emb = self.embedding(x_shifted)
                # Causal mask for only t+1 tokens
                mask = torch.triu(torch.ones(t + 1, t + 1, device=device), diagonal=1).bool()
                out = self.transformer(emb[:, :t + 1, :], mask=mask)
                logits = self.to_logits(out)  # (B, t+1, n_tokens)
                probs = torch.softmax(logits[:, -1, :], dim=-1)  # (B, n_tokens)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
                x_gen[:, t] = next_token
        return x_gen.view(B, h, w).to(torch.long)
