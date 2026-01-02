import jax
import jax.numpy as jnp
import equinox as eqx
import optax as tx
from typing import NamedTuple, Any
from jaxtyping import Key

# todo: Minimal Viable Train State to expand
class TrainState(NamedTuple): # Maybe change to eqx.Module if needed
    step: int
    params: eqx.Module
    opt_state: tx.OptState
    key: Key # Stores the reproducible seed

    @classmethod
    def create(cls, model: eqx.Module, optimizer: tx.GradientTransformationExtraArgs, key: jax.random.PRNGKey):
        """
        Create a trainstate that handles the initializes the optimizer and
        tracks the model's parameters.
        
        model - The model to track.
        optimizer - Description
        key - A random key
        """
        # Using the key for reproducibility:
        key, subkey = jax.random.split(key, 2)
        params = eqx.filter(model, eqx.is_array) # Getting the leaves
        opt_state = optimizer.init(params) # initializing based on params
        return cls(step=0, params=params, opt_state=opt_state, key=subkey)

def apply_gradients(state: TrainState, grads, optimizer: tx.GradientTransformationExtraArgs):
    updates, new_opt_state = optimizer.update(grads, state.opt_state, state.params)
    new_params = tx.apply_updates(state.params, updates)
    
    # Returning the updated states:
    return state._replace(
        step = state.step + 1,
        params=new_params,
        opt_state=new_opt_state
    )
