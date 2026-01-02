import jax
import equinox as eqx
import optax as tx
from .core import apply_gradients

def make_train_step(loss_fn: function, optimizer: tx.GradientTransformationExtraArgs):
    """Factory function to create a JIT-compiled training step."""

    @eqx.filter_jit
    def train_step(state, batch):
        key, subkey = jax.random.split(state.key)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(state.params, batch, subkey)
        new_state = apply_gradients(state, grads, optimizer)

        # Ensuring that the key is updated (Optax style):
        new_state = new_state._replace(key=key)
        return new_state, loss
    
    return train_step