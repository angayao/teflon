import haiku as hk
import jax 
import jax.numpy as jnp 

class GRU(hk.Module):
    def __init__(self, config):
        super().__init__()
        self.gru = hk.DeepRNN([hk.GRU(config.embed_dim)])
        self.mlp = hk.nets.MLP(output_sizes=[config.embed_dim * 4])
        self.fc = hk.Linear(config.embed_dim)
        self.fc_out = hk.Linear(config.horizon)

    def __call__(self, x):
        initial_state = self.gru.initial_state(x.shape[0])  # Batch size
        x, _ = hk.dynamic_unroll(self.gru, x, initial_state, time_major=False)
        x = self.mlp(x)
        x = jnp.reshape(x, (x.shape[0], -1))
        x = self.fc(x)
        x = self.fc_out(x)
        return x
