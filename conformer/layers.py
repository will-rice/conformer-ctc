"""
Conformer Layers
Adapted from:
https://huggingface.co/transformers/master/model_doc/transformerxl.html#tftransfoxlmodel
"""
import math
from typing import Any, Optional

import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras import layers


class SubsamplingConv1D(layers.Layer):
    """Subsampling Convolution Layer.
    Downsamples inputs from 10ms to 40ms which improves performance. Most open-source
    implementations use Conv2D, but we are using Conv1D.
    Default parameters from https://arxiv.org/abs/1909.06317.
    """

    def __init__(
        self,
        filters: int = 256,
        kernel_size: int = 3,
        strides: int = 2,
        activation: str = "relu",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._filters = filters
        self._kernel_size = kernel_size
        self._strides = strides
        self._activation = activation

        self.conv_1 = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            activation=activation,
        )
        self.conv_2 = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            activation=activation,
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Forward Pass."""
        out = self.conv_1(inputs)
        out = self.conv_2(out)
        return out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self._filters,
                "kernel_size": self._kernel_size,
                "strides": self._strides,
                "activation": self._activation,
            }
        )
        return config


class GatedLinearUnit(layers.Layer):
    """GLU Activation."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def call(self, inputs: Tensor) -> Tensor:
        """Forward Pass."""
        linear, gated = tf.split(value=inputs, num_or_size_splits=2, axis=-1)
        return linear * tf.nn.sigmoid(gated)


class DepthwiseConv1D(layers.Layer):
    """Depthwise Convolution."""

    def __init__(self, kernel_size: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.conv = layers.DepthwiseConv2D(kernel_size=(kernel_size, 1), padding="same")

    def call(self, inputs: Tensor) -> Tensor:
        """Forward Pass."""
        out = tf.expand_dims(inputs, -1)
        out = self.conv(out)
        out = tf.squeeze(out, -1)
        return out


class ConvolutionBlock(layers.Layer):
    """Conformer Convolution Block."""

    def __init__(
        self,
        filters: int,
        depthwise_kernel_size: int = 32,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-8,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.layer_norm = layers.LayerNormalization(epsilon=layer_norm_eps)
        self.conv_1 = layers.Conv1D(filters=2 * filters, kernel_size=1, padding="valid")
        self.glu = GatedLinearUnit()
        self.depthwise = DepthwiseConv1D(depthwise_kernel_size)
        self.batch_norm = layers.BatchNormalization()
        self.activation = layers.Activation("swish")
        self.conv_2 = layers.Conv1D(filters=filters, kernel_size=1, padding="valid")
        self.dropout = layers.Dropout(dropout)

    def call(self, inputs: Tensor, training: bool = False) -> Tensor:
        """Forward Pass."""
        out = self.layer_norm(inputs)
        out = self.conv_1(out)
        out = self.glu(out)
        out = self.depthwise(out)
        out = self.batch_norm(out, training=training)
        out = self.activation(out)
        out = self.conv_2(out)
        out = self.dropout(out, training=training)
        return out + inputs


class TFPositionwiseFF(layers.Layer):
    """Feedforward Layer."""

    def __init__(
        self,
        d_model: int,
        d_inner: int,
        dropout: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self._d_model = d_model
        self._d_inner = d_inner
        self._dropout = dropout

        self.fc_1 = layers.Dense(d_inner)
        self.activation = layers.Activation("swish")
        self.dropout_1 = layers.Dropout(dropout)

        self.fc_2 = layers.Dense(d_model)
        self.dropout_2 = layers.Dropout(dropout)
        self.layer_norm = layers.LayerNormalization()

    def call(self, inputs: Tensor, training: bool = False) -> Tensor:
        """Forward Pass."""
        out = self.layer_norm(inputs)
        out = self.fc_1(out)
        out = self.activation(out)
        out = self.dropout_1(out, training=training)
        out = self.fc_2(out)
        out = self.dropout_2(out, training=training)
        out = out * 0.5
        return out + inputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_model": self._d_model,
                "d_inner": self._d_inner,
                "dropout": self._dropout,
            }
        )
        return config


class RelativePositionEmbedding(layers.Layer):
    """Creates a positional embedding.
    This layer calculates the position encoding as a mix of sine and cosine
    functions with geometrically increasing wavelengths. Defined and formulized in
     "Attention is All You Need", section 3.5.
    (https://arxiv.org/abs/1706.03762).
    Args:
      hidden_size: Size of the hidden layer.
      min_timescale: Minimum scale that will be applied at each position
      max_timescale: Maximum scale that will be applied at each position.
    """

    def __init__(
        self,
        hidden_size: int,
        min_timescale: float = 1.0,
        max_timescale: float = 1.0e4,
        **kwargs,
    ):

        super().__init__(**kwargs)
        self._hidden_size = hidden_size
        self._min_timescale = min_timescale
        self._max_timescale = max_timescale

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self._hidden_size,
                "min_timescale": self._min_timescale,
                "max_timescale": self._max_timescale,
            }
        )
        return config

    def call(self, inputs, length=None):
        """Forward Pass."""
        if inputs is None and length is None:
            raise ValueError(
                "If inputs is None, `length` must be set in "
                "RelativePositionEmbedding()."
            )
        if inputs is not None:
            input_shape = tf.shape(inputs)
            if length is not None and length != input_shape[1]:
                raise ValueError(
                    "If inputs is not None, `length` must equal to input_shape[1]."
                )
            length = input_shape[1]
        position = tf.cast(tf.range(length), tf.float32)
        num_timescales = self._hidden_size // 2
        min_timescale, max_timescale = self._min_timescale, self._max_timescale
        log_timescale_increment = math.log(
            float(max_timescale) / float(min_timescale)
        ) / (tf.cast(num_timescales, tf.float32) - 1)
        inv_timescales = min_timescale * tf.exp(
            tf.cast(tf.range(num_timescales), tf.float32) * -log_timescale_increment
        )
        scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
        position_embeddings = tf.concat(
            [tf.sin(scaled_time), tf.cos(scaled_time)], axis=1
        )
        return position_embeddings


class RelativeMultiHeadAttention(layers.Layer):
    """Transformer-XL Attention."""

    def __init__(
        self, d_model: int, num_heads: int, dropout: float = 0.0, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self._d_model = d_model
        self._d_head = d_model // num_heads
        self._num_heads = num_heads
        self._sqrt_dim = math.sqrt(d_model)

        self.query_proj = layers.Dense(d_model)
        self.key_proj = layers.Dense(d_model)
        self.value_proj = layers.Dense(d_model)
        self.pos_proj = layers.Dense(d_model, use_bias=False)

        self.dropout = layers.Dropout(dropout)

        self.out_proj = layers.Dense(d_model)

    def build(self, input_shape: tf.TensorShape) -> None:
        """Build Layer."""
        self.u_bias = self.add_weight(
            name="u_bias",
            shape=(self._num_heads, self._d_head),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.v_bias = self.add_weight(
            name="v_bias",
            shape=(self._num_heads, self._d_head),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(
        self,
        inputs: Tensor,
        pos_embedding: Tensor,
        masks: Optional[Tensor] = None,
        training: bool = False,
    ) -> None:
        """Forward Pass."""
        batch_size = tf.shape(inputs)[0]

        query = self.query_proj(inputs)
        key = self.key_proj(inputs)
        value = self.value_proj(inputs)

        query = tf.reshape(query, (batch_size, -1, self._num_heads, self._d_head))
        key = tf.reshape(key, (batch_size, -1, self._num_heads, self._d_head))
        value = tf.reshape(value, (batch_size, -1, self._num_heads, self._d_head))

        pos_embedding = self.pos_proj(pos_embedding)
        pos_embedding = tf.reshape(
            pos_embedding, (batch_size, -1, self._num_heads, self._d_head)
        )
        content = query + self.u_bias
        content_score = tf.matmul(content, tf.transpose(key, (0, 1, 3, 2)))
        pos_score = tf.matmul(
            query + self.v_bias, tf.transpose(pos_embedding, (0, 1, 3, 2))
        )
        pos_score = self._rel_shift(pos_score)
        score = (content_score + pos_score) / self._sqrt_dim

        if masks is not None:
            masks = masks[:, :, tf.newaxis, tf.newaxis]
            score = tf.where(tf.cast(masks, dtype=tf.bool), score, 1e-9)

        attn = tf.nn.softmax(score, -1)
        attn = self.dropout(attn, training=training)
        attn = tf.transpose(tf.matmul(attn, value), (0, 2, 1, 3))
        attn = tf.reshape(attn, (batch_size, -1, self._d_model))
        out = self.out_proj(attn)
        return out

    @staticmethod
    def _rel_shift(inputs: Tensor) -> Tensor:
        input_shape = tf.shape(inputs)

        inputs = tf.pad(inputs, [[0, 0], [1, 0], [0, 0], [0, 0]])
        inputs = tf.reshape(
            inputs, [input_shape[1] + 1, input_shape[0], input_shape[2], input_shape[3]]
        )
        inputs = tf.slice(inputs, [1, 0, 0, 0], [-1, -1, -1, -1])
        inputs = tf.reshape(inputs, input_shape)
        return inputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_model": self._d_model,
                "d_head": self._d_head,
                "num_heads": self._num_heads,
                "sqrt_dim": self._sqrt_dim,
            }
        )
        return config


class MultiHeadedSelfAttention(layers.Layer):
    """MHSA."""

    def __init__(
        self,
        d_model: int,
        num_attention_heads: int,
        attention_dropout: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._d_model = d_model
        self._num_attention_heads = num_attention_heads
        self._attention_dropout = attention_dropout

        self.positional_encoding = RelativePositionEmbedding(d_model)
        self.layer_norm = layers.LayerNormalization()
        self.attention = RelativeMultiHeadAttention(
            d_model=d_model, num_heads=num_attention_heads, dropout=attention_dropout
        )
        self.dropout = layers.Dropout(attention_dropout)

    def call(
        self,
        inputs: Tensor,
        attention_mask: Optional[Tensor] = None,
        training: bool = False,
    ) -> Tensor:
        """Forward Pass."""
        batch_size = tf.shape(inputs)[0]

        pos_embedding = self.positional_encoding(inputs)[tf.newaxis, :, :]
        pos_embedding = tf.tile(pos_embedding, (batch_size, 1, 1))

        out = self.layer_norm(inputs)
        out = self.attention(
            out, pos_embedding=pos_embedding, masks=attention_mask, training=training
        )
        out = self.dropout(out, training=training)

        return out + inputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_model": self._d_model,
                "num_attention_heads": self._num_attention_heads,
                "attention_dropout": self._attention_dropout,
            }
        )
        return config


class ConformerBlock(layers.Layer):
    """Conformer Block."""

    def __init__(
        self,
        d_model: int = 144,
        num_attention_heads: int = 4,
        attention_dropout: float = 0.0,
        feed_forward_expansion_factor: int = 4,
        dropout: float = 0.1,
        depthwise_kernel_size: int = 32,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._d_model = d_model
        self._num_attention_heads = num_attention_heads
        self._attention_dropout = attention_dropout
        self._feed_forward_expansion_factor = feed_forward_expansion_factor
        self._dropout = dropout
        self._depthwise_kernel_size = depthwise_kernel_size

        self.feed_forward_1 = TFPositionwiseFF(
            d_model=d_model,
            d_inner=d_model * feed_forward_expansion_factor,
            dropout=dropout,
        )
        self.attention = MultiHeadedSelfAttention(
            d_model=d_model,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
        )
        self.conv_block = ConvolutionBlock(
            filters=d_model,
            depthwise_kernel_size=depthwise_kernel_size,
            dropout=dropout,
        )
        self.feed_forward_2 = TFPositionwiseFF(
            d_model=d_model,
            d_inner=d_model * feed_forward_expansion_factor,
            dropout=dropout,
        )

        self.layer_norm = layers.LayerNormalization()

    def call(
        self,
        inputs: Tensor,
        attention_mask: Optional[Tensor] = None,
        training: Optional[bool] = False,
    ) -> Tensor:
        """Forward Pass."""
        out = self.feed_forward_1(inputs, training=training)
        out = self.attention(out, attention_mask=attention_mask, training=training)
        out = self.conv_block(out, training=training)
        out = self.feed_forward_2(out, training=training)
        out = self.layer_norm(out)
        return out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_model": self._d_model,
                "num_attention_heads": self._num_attention_heads,
                "attention_dropout": self._attention_dropout,
                "feed_forward_expansion_factor": self._feed_forward_expansion_factor,
                "dropout": self._dropout,
                "depthwise_kernel_size": self._depthwise_kernel_size,
            }
        )
        return config


class ConformerEncoder(layers.Layer):
    """Conformer Encoder."""

    def __init__(
        self,
        units: int = 144,
        num_layers: int = 16,
        num_attention_heads: int = 4,
        feed_forward_expansion_factor: int = 4,
        encoder_dropout: float = 0.1,
        attention_dropout: float = 0.1,
        depthwise_kernel_size: int = 32,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._units = units
        self._num_layers = num_layers
        self._num_attention_heads = num_attention_heads
        self._feed_forward_expansion_factor = feed_forward_expansion_factor
        self._encoder_dropout = encoder_dropout
        self._attention_dropout = attention_dropout
        self._depthwise_kernel_size = depthwise_kernel_size

        self.subsampling_layer = SubsamplingConv1D(filters=units)
        self.linear_proj = layers.Dense(units)
        self.dropout = layers.Dropout(encoder_dropout)
        self._layers = [
            ConformerBlock(
                d_model=units,
                num_attention_heads=num_attention_heads,
                feed_forward_expansion_factor=feed_forward_expansion_factor,
                depthwise_kernel_size=depthwise_kernel_size,
                dropout=encoder_dropout,
                attention_dropout=attention_dropout,
            )
            for _ in range(num_layers)
        ]

    def call(
        self,
        inputs: Tensor,
        attention_mask: Optional[Tensor] = None,
        training: Optional[bool] = False,
    ) -> Tensor:
        """Forward Pass."""
        out = self.subsampling_layer(inputs)
        out = self.linear_proj(out)
        out = self.dropout(out, training=training)

        for layer in self._layers:
            out = layer(out, attention_mask=attention_mask, training=training)

        return out

    def get_config(self):
        """Layer configuration for serialization."""
        config = super().get_config()
        config.update(
            {
                "units": self._units,
                "num_layers": self._num_layers,
                "num_attention_heads": self._num_attention_heads,
                "feed_forward_expansion_factor": self._feed_forward_expansion_factor,
                "encoder_dropout": self._encoder_dropout,
                "attention_dropout": self._attention_dropout,
                "depthwise_kernel_size": self._depthwise_kernel_size,
            }
        )
        return config
