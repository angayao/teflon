import jax
import haiku as hk
import jax.numpy as jnp


class GAttention(hk.Module):
    def __init__(self, emb_dim: int, num_heads: int = 1, name=None):
        super().__init__(name=name)
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        batch_size, seq_len, _ = x.shape

        # (B, T, H*D_head)
        qk_proj = hk.Linear(self.num_heads * self.head_dim)(x)
        qk = jnp.reshape(qk_proj, (batch_size, seq_len,
                         self.num_heads, self.head_dim))
        qk = jnp.transpose(qk, (0, 2, 1, 3))  # (B, H, T, D_head)

        # Compute pairwise distances in each head's subspace
        x_diff = qk[:, :, :, None, :] - \
            qk[:, :, None, :, :]  # (B, H, T, T, D_head)
        d_squared = jnp.sum(x_diff ** 2, axis=-1) + 1e-6  # (B, H, T, T)

        # Learnable masses per head and token
        mass = hk.Linear(self.num_heads)(x)  # (B, T, H)
        mass = jax.nn.softplus(mass)  # Ensure positivity
        mass = jnp.transpose(mass, (0, 2, 1))  # (B, H, T)

        # Compute gravitational attention scores
        mass_i = mass[:, :, :, None]  # (B, H, T, 1)
        mass_j = mass[:, :, None, :]  # (B, H, 1, T)
        attn_scores = (mass_i * mass_j) / d_squared  # (B, H, T, T)

        # Softmax normalization
        attn_weights = jax.nn.softmax(attn_scores, axis=-1)  # (B, H, T, T)

        # Project values and apply attention
        # (B, T, H*D_head)
        v_proj = hk.Linear(self.num_heads * self.head_dim)(x)
        v = jnp.reshape(v_proj, (batch_size, seq_len,
                        self.num_heads, self.head_dim))
        v = jnp.transpose(v, (0, 2, 1, 3))  # (B, H, T, D_head)

        # Apply attention weights
        out = jnp.einsum("bhij,bhjd->bhid", attn_weights,
                         v)  # (B, H, T, D_head)
        out = jnp.transpose(out, (0, 2, 1, 3))  # (B, T, H, D_head)
        out = jnp.reshape(
            out, (batch_size, seq_len, self.emb_dim))  # (B, T, D)
        return out


class GRAN(hk.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.proj = hk.Sequential([
            hk.Linear(config.embed_dim),
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
            jax.nn.gelu
        ])
        self.conv = hk.Sequential([
            hk.Conv2D(
                output_channels=config.embed_dim * 2,
                kernel_shape=4,
                stride=1,
                padding='SAME',
                data_format='NHWC'
            ),
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
            jax.nn.gelu
        ])

        self.gran_layers = []
        self.norms = []
        for _ in range(config.num_layers):
            self.gran_layers.append(GAttention(
                emb_dim=config.embed_dim * 2,
                num_heads=config.num_heads,
            ))
            self.norms.append(hk.LayerNorm(
                axis=-1,
                create_scale=True,
                create_offset=True
            ))

        self.mlp = hk.Sequential([
            hk.Linear(config.embed_dim * 4),
            jax.nn.gelu,
            hk.Linear(config.embed_dim * 2),
            lambda x: hk.dropout(hk.next_rng_key(), config.dropout_rate, x) if (
                config.dropout_rate > 0) else x
        ])

        self.fc = hk.Sequential([
            hk.Flatten(),
            hk.Linear(config.embed_dim * 2 * config.window_size),
            jax.nn.gelu,
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        ])

        self.fc_out = hk.Linear(config.horizon)
        self.dropout_rate = config.dropout_rate

    def __call__(self, x, is_training=True):
        x = self.proj(x)
        x = self.conv(x)
        if is_training and self.dropout_rate > 0:
            x = hk.dropout(hk.next_rng_key(), self.dropout_rate, x)
        for i in range(self.config.num_layers):
            x_attn = self.gran_layers[i](x)
            x = x + x_attn
            x = self.norms[i](x)
            x = jax.nn.gelu(x)

            if is_training and self.dropout_rate > 0:
                x = hk.dropout(hk.next_rng_key(), self.dropout_rate, x)
        x_mlp = self.mlp(x)
        x = x + x_mlp
        x = self.fc(x)
        if is_training and self.dropout_rate > 0:
            x = hk.dropout(hk.next_rng_key(), self.dropout_rate, x)
        x = self.fc_out(x)
        return x


# class GRAN(hk.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.proj = hk.Linear(config.embed_dim)
#         self.conv = hk.Conv2D(output_channels=config.embed_dim * 2, kernel_shape=4)
#         self.gran_layers = [GAttention(
#             emb_dim=config.embed_dim * 2,
#             num_heads=config.num_heads
#         ) for _ in range(config.num_layers)]
#         self.norms = [hk.LayerNorm(
#             axis=-1,
#             create_scale=True,
#             create_offset=True
#         ) for _ in range(config.num_layers)]
#         self.mlp = hk.nets.MLP(output_sizes=[config.embed_dim])
#         self.fc = hk.Linear(config.embed_dim * config.window_size)
#         self.fc_out = hk.Linear(config.horizon)
#
#     def __call__(self, x):
#         x = self.proj(x)
#         x = self.conv(x)
#         for i in range(self.config.num_layers):
#             x_now = self.gran_layers[i](x)
#             x_now = x_now + x
#             x_now = self.norms[i](x_now)
#             x = jax.nn.relu6(x_now)
#         x = self.mlp(x)
#         x = jnp.reshape(x, (x.shape[0], -1))
#         x = self.fc(x)
#         x = self.fc_out(x)
#         return x
a
