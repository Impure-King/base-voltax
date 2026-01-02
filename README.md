# Voltax

**A Functional Training Engine for JAX & Equinox**

Voltax is a minimalist deep learning wrapper designed to bridge the gap between raw JAX transformations (`jit`, `vmap`, `scan`) and high-level training loops. Unlike traditional OOP frameworks that rely on mutable internal state, Voltax treats the **Training State** (params, optimizer, RNG) as an immutable, purely functional data structure.

### Key Features
* **Pure State Management:** Encapsulates model parameters, optimizer state, and RNG keys into a single `TrainState` container.
* **Bit-wise Reproducibility:** Enforces explicit PRNG key splitting at every training step, ensuring deterministic execution across different hardware configurations.
* **JAX Native:** Fully compatible with `equinox.filter_jit` and `jax.lax.scan` for high-performance compilation without side effects. Used `optax` like syntax.

---