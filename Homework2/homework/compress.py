from pathlib import Path
from typing import cast

import numpy as np
import torch
from PIL import Image

from .autoregressive import Autoregressive
from .bsq import Tokenizer


class Compressor:
    def __init__(self, tokenizer: Tokenizer, autoregressive: Autoregressive):
        super().__init__()
        self.tokenizer = tokenizer
        self.autoregressive = autoregressive

    def compress(self, x: torch.Tensor) -> bytes:
        """
        Compress the image into a torch.uint8 bytes stream (1D tensor).

        Use Huffman coding instead of arithmetic coding.
        """
        # Tokenize the input image into patches
        tokens = self.tokenizer.tokenize(x)

        # Flatten tokens for Huffman coding
        flattened_tokens = tokens.flatten()

        # Compute frequency of each token
        unique, counts = torch.unique(flattened_tokens, return_counts=True)
        freq_dict = dict(zip(unique.tolist(), counts.tolist()))

        # Build Huffman tree and generate codes
        huffman_tree = self._build_huffman_tree(freq_dict)
        huffman_codes = self._generate_huffman_codes(huffman_tree)

        # Encode tokens using Huffman codes
        encoded = ''.join(huffman_codes[token.item()] for token in flattened_tokens)

        # Convert binary string to bytes
        compressed_bytes = int(encoded, 2).to_bytes((len(encoded) + 7) // 8, byteorder='big')

        return compressed_bytes

    def decompress(self, x: bytes) -> torch.Tensor:
        """
        Decompress a tensor into a PIL image.
        You may assume the output image is 150 x 100 pixels.
        """
        # Convert bytes back to binary string
        binary_string = ''.join(f'{byte:08b}' for byte in x)

        # Decode binary string using Huffman codes
        decoded_tokens = []
        current_code = ''
        for bit in binary_string:
            current_code += bit
            if current_code in self.huffman_codes_reverse:
                decoded_tokens.append(self.huffman_codes_reverse[current_code])
                current_code = ''

        # Reshape tokens back to the original patch structure
        tokens = torch.tensor(decoded_tokens).reshape(self.tokenizer.output_shape)

        # Decode the tokens back into the image
        image = self.tokenizer.decode(tokens)

        return image


def compress(tokenizer: Path, autoregressive: Path, image: Path, compressed_image: Path):
    """
    Compress images using a pre-trained model.

    tokenizer: Path to the tokenizer model.
    autoregressive: Path to the autoregressive model.
    images: Path to the image to compress.
    compressed_image: Path to save the compressed image tensor.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tk_model = cast(Tokenizer, torch.load(tokenizer, weights_only=False).to(device))
    ar_model = cast(Autoregressive, torch.load(autoregressive, weights_only=False).to(device))
    cmp = Compressor(tk_model, ar_model)

    x = torch.tensor(np.array(Image.open(image)), dtype=torch.uint8, device=device)
    cmp_img = cmp.compress(x.float() / 255.0 - 0.5)
    with open(compressed_image, "wb") as f:
        f.write(cmp_img)


def decompress(tokenizer: Path, autoregressive: Path, compressed_image: Path, image: Path):
    """
    Decompress images using a pre-trained model.

    tokenizer: Path to the tokenizer model.
    autoregressive: Path to the autoregressive model.
    compressed_image: Path to the compressed image tensor.
    images: Path to save the image to compress.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tk_model = cast(Tokenizer, torch.load(tokenizer, weights_only=False).to(device))
    ar_model = cast(Autoregressive, torch.load(autoregressive, weights_only=False).to(device))
    cmp = Compressor(tk_model, ar_model)

    with open(compressed_image, "rb") as f:
        cmp_img = f.read()

    x = cmp.decompress(cmp_img)
    img = Image.fromarray(((x + 0.5) * 255.0).clamp(min=0, max=255).byte().cpu().numpy())
    img.save(image)


if __name__ == "__main__":
    from fire import Fire

    Fire({"compress": compress, "decompress": decompress})
