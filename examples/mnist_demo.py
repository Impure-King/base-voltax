import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from voltax.core import TrainState
from voltax.loop import make_train_step

# Defining the basic MLP model:
class MLP(eqx.Module):
    layers: list
    def __init__(self, key):
        key, key1, key2 = jax.random.split(key, 3)
        self.layers = [
            eqx.nn.Linear(784, 512, key=key1),
            jax.nn.relu,
            eqx.nn.Linear(512, 10, key=key2)
        ]
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Dummy data (slated for replacement with tfds mnist):
def get_batch(batch_size=32):
    return jnp.ones((batch_size, 784)), jnp.zeros((batch_size,), dtype=jnp.int32)

# Basic loss_fn:
def loss_fn(model, batch):
    x, y = batch
    logits = jax.vmap(model)(x)
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

# Training Loop:
def main():
    print("Initializing Voltax Engine...")
    key = jax.random.PRNGKey(42)
    
    # Init Model & Optimizer
    model = MLP(key)
    optimizer = optax.adam(1e-3)
    
    # Init State (Voltax Core)
    state = TrainState.create(model, optimizer, seed=0)
    
    # JIT Compile the Step (Voltax Loop)
    train_step = make_train_step(loss_fn, optimizer)
    
    print("Starting Training Loop (Functional)...")
    for i in range(100):
        batch = get_batch()
        # The Pure Functional Update
        state, loss = train_step(state, batch)
        
        if i % 10 == 0:
            print(f"Step {i} | Loss: {loss:.4f} | RNG Key: {state.key[0]}")

if __name__ == "__main__":
    main()