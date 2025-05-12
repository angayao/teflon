#
# import haiku as hk
# import jax.numpy as jnp
# import jax
#
#
# class Transformer(hk.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.conv = hk.Conv1D(output_channels=config.embed_dim, kernel_shape=4)
#         self.attn_layers = [hk.MultiHeadAttention(
#             num_heads=config.num_heads,
#             key_size=config.embed_dim,
#             value_size=config.embed_dim,
#             model_size=config.embed_dim * 2,
#             w_init=hk.initializers.VarianceScaling(2.)
#         ) for _ in range(4)]
#         self.norms = [hk.LayerNorm(axis=1,create_scale=True, create_offset=True) for _ in range(4)]
#         self.mlp = hk.nets.MLP(output_sizes=[config.embed_dim])
#         self.fc = hk.Linear(config.embed_dim)
#         self.fc_out = hk.Linear(config.horizon)
#
#     def __call__(self, x):
#         x = self.conv(x)
#         for i in range(4):
#             x = self.attn_layers[i](x, x, x)
#             x = self.norms[i](x)
#         x = self.mlp(x)
#         x = jnp.reshape(x, (x.shape[0], -1))
#         x = self.fc(x)
#         x = self.fc_out(x)
#         return x
#
#
#
import haiku as hk
import jax.numpy as jnp
import numpy as np
import jax

class PositionalEncoding(hk.Module):
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len

    def __call__(self, x):
        position = jnp.arange(self.max_len)[:, None]
        div_term = jnp.exp(jnp.arange(0, self.embed_dim, 2) * (-np.log(10000.0) / self.embed_dim))
        pe = jnp.zeros((self.max_len, self.embed_dim))
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
        return x + pe[:x.shape[1]]

class TransformerEncoderLayer(hk.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim, dropout_rate=0.1, name=None):
        super().__init__(name=name)
        self.attn = hk.MultiHeadAttention(
            num_heads=num_heads,
            key_size=embed_dim // num_heads,
            model_size=embed_dim,
            w_init=hk.initializers.VarianceScaling(2.)
        )
        self.norm1 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self.norm2 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self.ffn = hk.Sequential([
            hk.Linear(ffn_dim), jax.nn.relu,
            hk.Linear(embed_dim)
        ])
        self.dropout_rate = dropout_rate

    def __call__(self, x, is_training):
        # Self-attention with residual and norm
        attn_out = self.attn(x, x, x)
        x = x + hk.dropout(hk.next_rng_key(), self.dropout_rate, attn_out) if is_training else x + attn_out
        x = self.norm1(x)

        # Feedforward with residual and norm
        ffn_out = self.ffn(x)
        x = x + hk.dropout(hk.next_rng_key(), self.dropout_rate, ffn_out) if is_training else x + ffn_out
        x = self.norm2(x)
        return x

class Transformer(hk.Module):
    def __init__(self, config, name=None):
        super().__init__(name=name)
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.ffn_dim = config.embed_dim
        self.num_layers = 4
        self.horizon = config.horizon
        self.dropout_rate = getattr(config, "dropout_rate", 0.1)

        self.input_proj = hk.Linear(self.embed_dim)  # Like token embedding
        self.pos_enc = PositionalEncoding(self.embed_dim)

        self.encoder_layers = [
            TransformerEncoderLayer(self.embed_dim, self.num_heads, self.ffn_dim, self.dropout_rate, name=f"encoder_layer_{i}")
            for i in range(self.num_layers)
        ]

        self.output_proj = hk.Linear(self.horizon)

    def __call__(self, x, is_training=True):
        """
        Args:
            x: shape (batch, seq_len, features)
        Returns:
            output: shape (batch, horizon)
        """
        x = self.input_proj(x)  # Project input to embed_dim
        x = self.pos_enc(x)

        for layer in self.encoder_layers:
            x = layer(x, is_training=is_training)

        x = jnp.mean(x, axis=1)  # Global average pooling
        x = self.output_proj(x)
        return x
