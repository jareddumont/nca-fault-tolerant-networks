# Neural Cellular Automata with Fault Tolerance

**What if neural networks could survive 95% of their weights being corrupted?**



Standard neural networks collapse under weight corruption. Our NCA-based approach recovers fully—not by repairing damage, but by being immune to it.

| Metric | NCA | Standard MLP |
|--------|-----|--------------|
| Baseline Accuracy | 97.1% | 97.2% |
| After 95% Bitflip Damage | **97.1%** | 4.4% |
| Recovery vs Random Chance | +87pp | -6pp |

*10 seeds, p < 0.0001*

---

## The Key Insight

We don't train neural network weights directly. Instead, we train a **Neural Cellular Automaton** to *generate* weights through local self-organization.

The NCA learns a **landscape of equivalent solutions** rather than a single correct state. Corrupt 95% of the cells? The dynamics converge back to a functional configuration—not the *same* configuration, but an *equivalent* one.

This is **damage immunity through solution degeneracy**, not self-repair in the traditional sense.



---

## Quick Start

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/nca-fault-tolerance.git
cd nca-fault-tolerance

# Install
pip install -r requirements.txt

# See it work
python demo.py
```

The demo trains a small NCA on MNIST, corrupts 95% of its state, and shows recovery in real-time.

---

## How It Works

### Architecture



1. **Hierarchical NCA Grid**: 3D grid of cells with local update rules
2. **Weight Projection**: Grid state → neural network weights
3. **Task Evaluation**: Generated weights solve MNIST classification
4. **Training**: Evolutionary optimization of NCA update rules

### The Self-Repair Effect

After damage, running additional NCA steps actively improves accuracy:

| State | Accuracy |
|-------|----------|
| Healthy baseline | 97.1% |
| Immediately after 95% damage | 68.8% |
| +15 NCA recovery steps | **97.1%** |

The **+28.3% recovery** from additional steps proves this is active self-organization, not just robust weights.

---

## Benchmark Results

### Damage Type Comparison



| Damage Type | NCA Recovery | MLP Recovery |
|-------------|--------------|--------------|
| 90% Bitflip | 97.2% | 9.8% |
| 95% Bitflip | 97.1% | 4.4% |
| 50% Zero-out | 97.3% | 52.1% |
| Gaussian Noise (σ=2) | 96.8% | 11.2% |

### Architecture Ablations



- **Grid size**: Larger grids (16³) improve fault tolerance
- **Hierarchy levels**: 2-3 levels optimal
- **Channel count**: 8+ channels sufficient
- **Recovery steps**: 15 steps for full recovery

---

## Why This Matters

### Applications

- **Edge computing**: Devices with unreliable memory
- **Spacecraft AI**: Radiation-induced bit flips
- **Neuromorphic hardware**: Inherently noisy substrates
- **Long-running deployments**: Graceful degradation over time

### What We're NOT Claiming

The fault tolerance is **task-specific**. An NCA trained on MNIST develops an MNIST-specific attractor basin. It won't magically generalize fault tolerance to unseen tasks.



This is a feature, not a bug—the NCA learns robust dynamics *for the task it's trained on*.

---

## Repository Structure

```
nca-fault-tolerance/
├── README.md
├── demo.py                      # Quick demonstration
├── requirements.txt
├── figures/                     # All paper figures
├── src/
│   ├── hierarchical_nca.py      # Core NCA architecture
│   ├── train_mnist.py           # Training script
│   └── test_nca_self_repair.py  # Benchmark suite
└── benchmarks/
    └── RESULTS.md               # Detailed statistics
```

---

## Run Your Own Benchmarks

```bash
# Full 10-seed benchmark (2-4 hours on GPU)
python src/test_nca_self_repair.py --seeds 10 --damage-levels 0.5 0.9 0.95

# Quick single-seed test (10 minutes)
python src/test_nca_self_repair.py --seeds 1 --damage-levels 0.95
```

This repository is an experimental research artifact exploring fault tolerance through generative dynamics. Results are reproducible but not claimed to generalize beyond evaluated tasks.

---

## Citation

If you use this work, please cite:

```bibtex
@software{nca_fault_tolerance_2024,
  author = {Jared Dumont},
  title = {Neural Cellular Automata with Fault Tolerance},
  year = {2025},
  url = {https://github.com/jareddumont/nca-fault-tolerant-networks}
}
```

---

## Related Work

- [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/) - Mordvintsev et al., 2020
- [Self-Organizing Neural Networks](https://arxiv.org/abs/2009.01397) - Randazzo et al., 2020
- [HyperNetworks](https://arxiv.org/abs/1609.09106) - Ha et al., 2017

---

## Contact

Questions? Open an issue or reach out at jareddumont1@gmail.com

---

## License

MIT
