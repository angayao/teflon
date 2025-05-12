import haiku as hk 
import jax.numpy as jnp 

class RNN(hk.Module):
    def __init__(self, config): 
        super().__init__()
        self.config = config
        self.rnn = hk.LSTM(hidden_size=config.embed_dim)
        self.fc = hk.Linear(config.embed_dim)
        self.fc_out = hk.Linear(config.horizon)

    def __call__(self, x):
        b, t, _ = x.shape 
        state = self.rnn.initial_state(b)
        out = x 
        for t in range(t):
            out, state = self.rnn(x[:, t, :], state)
        x = out 
        x = jnp.reshape(x, (x.shape[0], -1))
        x = self.fc(x)
        x = self.fc_out(x)
        return x 


