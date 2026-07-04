# Voltax

**A Functional Training Engine for JAX & Equinox**

Voltax is a minimalist deep learning wrapper designed to bridge the gap between raw JAX transformations (`jit`, `vmap`, `scan`) and high-level training loops. Unlike traditional OOP frameworks that rely on mutable internal state, Voltax treats the **Training State** (params, optimizer, RNG) as an immutable, purely functional data structure.

### Key Features
* **Pure State Management:** Encapsulates model parameters, optimizer state, and RNG keys into a single `TrainState` container.
* **Bit-wise Reproducibility:** Enforces explicit PRNG key splitting at every training step, ensuring deterministic execution across different hardware configurations.
* **JAX Native:** Fully compatible with `equinox.filter_jit` and `jax.lax.scan` for high-performance compilation without side effects. Used `optax` like syntax.

---

## Current Status

Early development. Core `TrainState` abstraction is in place. The immediate goal is to build out a complete training pipeline with data loading, checkpointing, logging, and evaluation — all under strict resource constraints.

### Current Milestone
- Implement end-to-end training loop with reproducibility guarantees
- Add checkpoint save/resume with atomic writes
- Basic experiment tracking (config + metrics logging)
- Profile memory and throughput on the RTX 4060

### Definition of Done
A researcher can clone this repo, run a single command, train a small model on a standard dataset, resume from checkpoint, and reproduce the run bit-for-bit given the same seed and config.

---

## How to Run

```bash
# Install dependencies (inside distrobox)
pip install -e .

# Run a training example
python -m voltax.examples.mnist

# Run with custom config
python -m voltax.train --config configs/mnist_basic.yaml
```

*(Commands will evolve as the project matures.)*

---

## Constraints

- **Hardware:** 12th gen Intel i7 CPU, Nvidia RTX 4060 (8 GB VRAM), 16 GB RAM, 128 GB storage
- **Environment:** Fedora Distrobox on Bazzite OS
- **Budget:** Up to 150 CAD Azure credits (on request only)
- **GitHub:** Personal free-tier limits apply
- **Non-goals:** Large-scale distributed training, multi-node orchestration, production deployment. This is a research infrastructure project for a single GPU.

---

## Project Structure

```
voltax/            # Core library
examples/          # Training examples
configs/           # YAML configs for reproducible runs
tests/             # Test suite
.prompts/          # Agent prompt files (Akame, Sylvia, Ethan)
.github/           # Issue templates for pod subagents
.sandbox/          # Agent working directories (gitignored)
```

---

## Pod-of-Three

This project uses the Pod-of-Three pattern with OpenCode agents:
- **Akame** — Tech lead and mentor. Owns roadmap, standups, backlog, triage.
- **Sylvia** — Code reviewer. Rigorous, evidence-based reviews.
- **Ethan** — Design partner. Real-usage feedback and benchmarks.

See `opencode.jsonc` and `prompts/` for agent configurations.