import haiku as hk 
import jax 
import jax.numpy as jnp 


class MLP(hk.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp1 = hk.nets.MLP(output_sizes=[config.embed_dim * 2])
        self.mlp2 = hk.nets.MLP(output_sizes=[config.embed_dim])
        self.fc = hk.Linear(config.embed_dim)
        self.fc_out = hk.Linear(config.horizon)

    def __call__(self, x):
        x = self.mlp1(x) 
        x = self.mlp2(x) 
        x = jnp.reshape(x, (x.shape[0], -1))
        x = self.fc(x)
        x = jax.nn.relu6(x)
        x = self.fc_out(x)
        return x
