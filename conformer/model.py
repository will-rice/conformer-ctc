"""
Conformer: Convolution-augmented Transformer for Speech Recognition
https://arxiv.org/pdf/2005.08100.pdf
"""
from typing import Any, Optional

import tensorflow as tf
from tensorflow import Tensor

from conformer.config import ConformerConfig
from conformer.layers import ConformerEncoder


class ConformerForCTC(tf.keras.Model):
    """Conformer Model."""

    def __init__(self, config: ConformerConfig, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.encoder = ConformerEncoder(
            units=config.encoder_units,
            num_layers=config.encoder_num_layers,
            num_attention_heads=config.num_attention_heads,
            feed_forward_expansion_factor=config.feed_forward_expansion_factor,
            encoder_dropout=config.encoder_dropout,
            attention_dropout=config.attention_dropout,
            depthwise_kernel_size=config.depthwise_kernel_size,
        )
        self.head = tf.keras.layers.Dense(units=config.vocab_size)

    def call(
        self,
        inputs: Tensor,
        attention_mask: Optional[Tensor] = None,
        training: bool = False,
    ) -> Tensor:
        """Forward Pass."""
        out = self.encoder(inputs, attention_mask=attention_mask, training=training)
        out = self.head(out)
        return out

    def recognize(self, inputs: Tensor, lengths: int, greedy: bool = False) -> Tensor:
        """Inference Pass."""
        input_shape = tf.shape(inputs)
        lengths = lengths // 4 - 1

        attention_mask = tf.sequence_mask(lengths, maxlen=input_shape[1] // 4 - 1)
        logits = self.call(inputs, attention_mask=attention_mask, training=False)
        logits = tf.transpose(logits, (1, 0, 2))

        if greedy:
            decoded, _ = tf.nn.ctc_greedy_decoder(logits, sequence_length=lengths)
        else:
            decoded, _ = tf.nn.ctc_beam_search_decoder(
                logits, sequence_length=lengths, beam_width=10
            )
        return decoded[0]
