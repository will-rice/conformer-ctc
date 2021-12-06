"""
Conformer Config.
+------------------+---------------+---------------+---------------+
| Model            | Conformer (S) | Conformer (M) | Conformer (L) |
+------------------+---------------+---------------+---------------+
| Num Params (M)   | 10.3          | 30.7          | 118.8         |
+------------------+---------------+---------------+---------------+
| Encoder Layers   | 16            | 16            | 17            |
+------------------+---------------+---------------+---------------+
| Encoder Dim      | 144           | 256           | 512           |
+------------------+---------------+---------------+---------------+
| Attention Heads  | 4             | 4             | 8             |
+------------------+---------------+---------------+---------------+
| Conv Kernel Size | 32            | 32            | 32            |
+------------------+---------------+---------------+---------------+
| Decoder Layers   | 1             | 1             | 1             |
+------------------+---------------+---------------+---------------+
| Decoder Dim      | 320           | 640           | 640           |
+------------------+---------------+---------------+---------------+
"""
from dataclasses import dataclass


@dataclass
class ConformerConfig:
    """Conformer Config."""

    vocab_size: int = 32
    # Encoder
    encoder_num_layers: int = 16
    encoder_units: int = 144
    encoder_dropout: float = 0.1
    num_attention_heads: int = 4
    feed_forward_expansion_factor: int = 4
    attention_dropout: float = 0.1
    depthwise_kernel_size: int = 32
    # Decoder
    decoder_units: int = 320
    decoder_num_layers: int = 1
