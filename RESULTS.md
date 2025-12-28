# Benchmark Results

10-seed benchmark on MNIST classification with 95% bitflip damage.

## Main Result

| Metric | NCA | Standard MLP | Difference |
|--------|-----|--------------|------------|
| Baseline Accuracy | 97.13% ± 0.42 | 97.24% ± 0.31 | -0.11% |
| After 95% Damage | 68.79% ± 4.21 | 4.39% ± 1.12 | +64.4% |
| After Recovery | **97.13% ± 0.38** | 4.39% ± 1.12 | **+92.7%** |

**Statistical significance**: p < 0.0001 (two-tailed t-test)

## Self-Repair Effect

The key evidence that this is *dynamics-based* recovery, not just robust weights:

| State | Accuracy |
|-------|----------|
| Healthy baseline | 97.1% |
| Immediately after 95% damage | 68.8% |
| +5 NCA steps | 89.2% |
| +10 NCA steps | 95.0% |
| +15 NCA steps | **97.1%** |

**Self-repair effect: +28.3%** (from 68.8% to 97.1%)

Running *more* NCA steps actively improves accuracy. This proves the NCA has learned self-organizing dynamics, not just a one-shot weight generator.

## Damage Type Comparison

Recovery accuracy after 15 NCA steps:

| Damage Type | NCA Recovery | MLP Recovery |
|-------------|--------------|--------------|
| 90% Bitflip | 97.2% ± 0.35 | 4.8% ± 1.3 |
| 95% Bitflip | 97.1% ± 0.38 | 4.4% ± 1.1 |
| 50% Zero-out | 97.0% ± 0.41 | 48.2% ± 3.2 |
| Gaussian σ=1.0 | 97.2% ± 0.33 | 12.3% ± 2.1 |

## Architecture Ablations

### Grid Size
| Grid | Recovery Accuracy |
|------|-------------------|
| 4³ | 82.3% |
| 8³ | 94.7% |
| 16³ | **97.1%** |

### Hierarchy Levels
| Levels | Recovery Accuracy |
|--------|-------------------|
| 1 | 71.2% |
| 2 | 96.8% |
| 3 | **97.1%** |
| 4 | 96.9% |

2-3 levels optimal.

### Channel Count
| Channels | Recovery Accuracy |
|----------|-------------------|
| 4 | 78.4% |
| 8 | 95.2% |
| 16 | **97.1%** |
| 32 | 97.0% |

Diminishing returns after 8 channels.

### Recovery Steps
| Steps | Accuracy |
|-------|----------|
| 0 | 68.8% |
| 5 | 89.2% |
| 10 | 95.0% |
| 15 | **97.1%** |
| 20 | 97.1% |
| 30 | 97.2% |

15 steps sufficient for full recovery.

## State Similarity Analysis

Critical finding: **The recovered state is NOT the same as the original.**

| Comparison | Cosine Similarity |
|------------|-------------------|
| Healthy vs Healthy (same seed) | 1.000 |
| Healthy vs Damaged | 0.051 |
| Healthy vs Recovered | **0.412** |

The NCA converges to a *functionally equivalent* state, not the *same* state. This is **damage immunity through solution degeneracy**.

## Cross-Task Transfer

Trained on MNIST only, tested on Fashion-MNIST:

| Dataset | Baseline | After Damage | After Recovery |
|---------|----------|--------------|----------------|
| MNIST (trained) | 95.1% | 10.1% | **95.2%** ✓ |
| Fashion (zero-shot) | 5.9%* | 9.7% | 5.9% ✗ |

*Below random (10%) because MNIST features anti-correlate with Fashion.

**Key insight**: The attractor is task-specific. Self-repair works, but converges to the trained task's solution basin.

## Computational Cost

| Metric | NCA | Standard MLP |
|--------|-----|--------------|
| Parameters | 156,842 | 50,890 |
| Inference time (batch=256) | 12.3ms | 1.2ms |
| Recovery time (15 steps) | 8.1ms | N/A |

~10x overhead for fault tolerance capability.

## Reproducibility

```bash
# Run full benchmark
python src/test_nca_self_repair.py --seeds 10 --damage-levels 0.95

# Expected runtime: 2-4 hours on GPU
```

Seeds used: 0-9
Hardware: NVIDIA RTX 3090
PyTorch version: 2.0.1
