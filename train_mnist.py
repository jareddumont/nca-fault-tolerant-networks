#!/usr/bin/env python3
"""
MNIST Training for Hierarchical NCA

Tests if the hierarchical CA can learn real-world image recognition.
Uses 4-level architecture scaled up: 32³→16³→8³→4³

This is the "Tier 2" test - if we get 70%+ accuracy, we have something real.
"""

import argparse
import time
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hierarchical_nca import (
    HierarchicalNCA,
    HierarchyConfig,
    WEIGHTS_PER_LEVEL,
    print_config_info,
)

# MNIST-compatible 4-level config - matches threshold training for transfer learning
# 16³→8³→4³→2³ = 4,680 cells (same as 100% threshold model)
CONFIG_MNIST = HierarchyConfig(
    level_sizes=[16, 8, 4, 2],
    input_size=16
)


def download_mnist(data_dir="./mnist_data"):
    """Download MNIST dataset."""
    try:
        from torchvision import datasets, transforms

        transform = transforms.Compose([
            transforms.Resize((16, 16)),  # Resize 28x28 to 16x16
            transforms.ToTensor(),
        ])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

        return train_dataset, test_dataset
    except ImportError:
        print("ERROR: torchvision required for MNIST")
        print("Install with: pip install torchvision")
        sys.exit(1)


class MNISTHierarchicalNCA(nn.Module):
    """
    Hierarchical NCA with classification head for MNIST.

    The CA processes the image, then a linear layer classifies.
    """

    def __init__(self, config: HierarchyConfig, device: torch.device):
        super().__init__()
        self.config = config
        self.device = device
        self.nca = HierarchicalNCA(config, device)

        # Classification head: top level has 2³=8 cells × 4 channels = 32 features
        top_size = config.level_sizes[-1]
        top_features = (top_size ** 3) * 4  # 4 channels
        self.classifier = nn.Linear(top_features, 10).to(device)

    def forward(self, x, steps=15):
        """
        x: (batch, 1, 16, 16) grayscale images
        returns: (batch, 10) class logits
        """
        batch_size = x.shape[0]

        # Initialize CA states
        states = self.nca.initialize_state(batch_size)

        # Set input (squeeze channel dim, threshold to binary)
        input_pattern = (x.squeeze(1) > 0.5).float()
        self.nca.set_input(states, input_pattern)

        # Run CA steps
        for _ in range(steps):
            states = self.nca.step(states)

        # Get top level features
        top_state = states[-1]  # (batch, 2, 2, 2, 4)
        features = top_state.reshape(batch_size, -1)  # (batch, 32)

        # Classify
        logits = self.classifier(features)
        return logits

    def set_nca_weights(self, weights: np.ndarray):
        """Set NCA weights from numpy array."""
        self.nca.set_weights(weights)


class MNISTEvolutionaryTrainer:
    """Evolutionary trainer for MNIST classification."""

    def __init__(self, config: HierarchyConfig, device: torch.device,
                 population_size=30, elite_count=5):
        self.config = config
        self.device = device
        self.population_size = population_size
        self.elite_count = elite_count

        # Create model for evaluation
        self.model = MNISTHierarchicalNCA(config, device)

        # Weight counts
        self.nca_weights = config.total_weights
        self.classifier_weights = (config.level_sizes[-1] ** 3) * 4 * 10 + 10  # weights + bias
        self.total_weights = self.nca_weights + self.classifier_weights

        print(f"NCA weights: {self.nca_weights}")
        print(f"Classifier weights: {self.classifier_weights}")
        print(f"Total weights: {self.total_weights}")

        # Population
        self.population = None
        self.fitness = None
        self.best_genome = None
        self.best_fitness = 0.0
        self.generation = 0

        # Pre-trained NCA support
        self.frozen_nca = False
        self.pretrained_nca_weights = None

        self.rng = np.random.default_rng()

    def load_pretrained_nca(self, path: str):
        """Load pre-trained NCA weights and freeze them."""
        data = np.load(path)
        genome = data['genome']
        # Extract just the NCA weights (first 7408)
        self.pretrained_nca_weights = genome[:self.nca_weights].copy()
        self.frozen_nca = True
        print(f"Loaded pre-trained NCA from: {path}")
        print(f"NCA weights frozen - only training classifier ({self.classifier_weights} weights)")

    def initialize_population(self):
        """Initialize population - with frozen NCA if pre-trained."""
        if self.frozen_nca and self.pretrained_nca_weights is not None:
            # Only randomize classifier weights
            self.population = np.zeros(
                (self.population_size, self.total_weights), dtype=np.float32
            )
            for i in range(self.population_size):
                # Copy frozen NCA weights
                self.population[i, :self.nca_weights] = self.pretrained_nca_weights
                # Random classifier weights
                self.population[i, self.nca_weights:] = self.rng.standard_normal(
                    self.classifier_weights
                ).astype(np.float32) * 0.3
        else:
            # Fully random population
            self.population = self.rng.standard_normal(
                (self.population_size, self.total_weights)
            ).astype(np.float32) * 0.3
        self.fitness = np.zeros(self.population_size)

    def set_weights(self, genome: np.ndarray):
        """Set model weights from genome."""
        # NCA weights
        self.model.nca.set_weights(genome[:self.nca_weights])

        # Classifier weights
        classifier_w = genome[self.nca_weights:self.nca_weights + self.classifier_weights - 10]
        classifier_b = genome[self.nca_weights + self.classifier_weights - 10:]

        with torch.no_grad():
            self.model.classifier.weight.copy_(
                torch.from_numpy(classifier_w.reshape(10, -1)).to(self.device)
            )
            self.model.classifier.bias.copy_(
                torch.from_numpy(classifier_b).to(self.device)
            )

    def evaluate(self, genome: np.ndarray, images: torch.Tensor,
                 labels: torch.Tensor, steps=15) -> float:
        """Evaluate a genome on a batch of images."""
        self.set_weights(genome)

        with torch.no_grad():
            logits = self.model(images, steps=steps)
            predictions = logits.argmax(dim=1)
            accuracy = (predictions == labels).float().mean().item()

        return accuracy

    def evaluate_population(self, images: torch.Tensor, labels: torch.Tensor, steps=15):
        """Evaluate entire population."""
        for i in range(self.population_size):
            self.fitness[i] = self.evaluate(self.population[i], images, labels, steps)

        best_idx = np.argmax(self.fitness)
        if self.fitness[best_idx] > self.best_fitness:
            self.best_fitness = self.fitness[best_idx]
            self.best_genome = self.population[best_idx].copy()
            return True
        return False

    def create_next_generation(self):
        """Create next generation via selection, crossover, mutation."""
        sorted_indices = np.argsort(self.fitness)[::-1]
        next_gen = np.zeros_like(self.population)

        # Elitism
        for i in range(self.elite_count):
            next_gen[i] = self.population[sorted_indices[i]].copy()

        # Rest via crossover and mutation
        for i in range(self.elite_count, self.population_size):
            # Tournament selection
            idx1 = self.rng.choice(self.population_size, size=3, replace=False)
            idx2 = self.rng.choice(self.population_size, size=3, replace=False)
            p1 = self.population[idx1[np.argmax(self.fitness[idx1])]]
            p2 = self.population[idx2[np.argmax(self.fitness[idx2])]]

            if self.frozen_nca:
                # Only crossover/mutate classifier weights
                child = p1.copy()  # Start with parent 1
                child[:self.nca_weights] = self.pretrained_nca_weights  # Keep frozen NCA

                # Crossover only classifier
                cls_mask = self.rng.random(self.classifier_weights) < 0.5
                child[self.nca_weights:] = np.where(
                    cls_mask,
                    p1[self.nca_weights:],
                    p2[self.nca_weights:]
                )

                # Mutation only classifier
                mut_mask = self.rng.random(self.classifier_weights) < 0.15  # Higher mutation rate
                child[self.nca_weights:] += mut_mask * self.rng.standard_normal(
                    self.classifier_weights
                ).astype(np.float32) * 0.3
            else:
                # Full crossover
                mask = self.rng.random(self.total_weights) < 0.5
                child = np.where(mask, p1, p2)

                # Full mutation
                mut_mask = self.rng.random(self.total_weights) < 0.1
                child += mut_mask * self.rng.standard_normal(self.total_weights).astype(np.float32) * 0.2

            next_gen[i] = child

        self.population = next_gen
        self.generation += 1

    def save_best(self, path: str):
        """Save best genome."""
        if self.best_genome is not None:
            np.savez(path,
                     genome=self.best_genome,
                     fitness=self.best_fitness,
                     generation=self.generation)


def main():
    parser = argparse.ArgumentParser(description="MNIST training for Hierarchical NCA")
    parser.add_argument('--generations', type=int, default=500)
    parser.add_argument('--population', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Images per evaluation')
    parser.add_argument('--steps', type=int, default=15)
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--test-interval', type=int, default=50,
                        help='Run test set evaluation every N generations')
    parser.add_argument('--pretrained-nca', type=str, default=None,
                        help='Path to pre-trained NCA weights (from threshold training)')
    args = parser.parse_args()

    print("=" * 70)
    print("  MNIST Training - Hierarchical NCA")
    print("  THE REAL TEST: Can we recognize handwritten digits?")
    print("=" * 70)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    config = CONFIG_MNIST
    print_config_info(config)

    # Load MNIST
    print("\nLoading MNIST dataset...")
    train_dataset, test_dataset = download_mnist()
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    # Create trainer
    trainer = MNISTEvolutionaryTrainer(
        config, device,
        population_size=args.population,
        elite_count=max(3, args.population // 10)
    )

    # Load pre-trained NCA if provided
    if args.pretrained_nca:
        trainer.load_pretrained_nca(args.pretrained_nca)

    trainer.initialize_population()

    # Output path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pretrain_suffix = "_pretrained" if args.pretrained_nca else ""
    output_path = f"mnist_hierarchical{pretrain_suffix}_{timestamp}.npz"

    print(f"\nTraining for {args.generations} generations...")
    print(f"Batch size: {args.batch_size}")
    print(f"CA steps: {args.steps}")
    print(f"Output: {output_path}")
    print()
    print("-" * 70)
    print(f"{'Gen':>6} | {'Train':>7} | {'Best':>7} | {'Test':>7} | {'Time':>6}")
    print("-" * 70)

    start_time = time.time()
    train_iter = iter(train_loader)

    try:
        for gen in range(args.generations):
            gen_start = time.time()

            # Get batch
            try:
                images, labels = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                images, labels = next(train_iter)

            images = images.to(device)
            labels = labels.to(device)

            # Evaluate and evolve
            is_new = trainer.evaluate_population(images, labels, args.steps)
            best_train = np.max(trainer.fitness)
            trainer.create_next_generation()

            gen_time = time.time() - gen_start

            # Test evaluation
            test_acc = None
            if gen % args.test_interval == 0 or gen == args.generations - 1:
                # Evaluate on test set
                test_images, test_labels = next(iter(test_loader))
                test_images = test_images.to(device)
                test_labels = test_labels.to(device)
                test_acc = trainer.evaluate(trainer.best_genome, test_images, test_labels, args.steps)

            # Logging
            if gen % args.log_interval == 0 or is_new or gen == args.generations - 1:
                test_str = f"{test_acc:6.1%}" if test_acc is not None else "   -  "
                marker = " << NEW" if is_new else ""
                print(f"{gen:6d} | {best_train:6.1%} | {trainer.best_fitness:6.1%} | {test_str} | {gen_time:5.1f}s{marker}")

            if is_new:
                trainer.save_best(output_path)

    except KeyboardInterrupt:
        print("\n\nTraining interrupted.")

    # Final evaluation on full test set
    print("\n" + "=" * 70)
    print("  Final Evaluation on Test Set")
    print("=" * 70)

    total_correct = 0
    total_samples = 0

    trainer.set_weights(trainer.best_genome)

    for test_images, test_labels in test_loader:
        test_images = test_images.to(device)
        test_labels = test_labels.to(device)

        with torch.no_grad():
            logits = trainer.model(test_images, steps=args.steps)
            predictions = logits.argmax(dim=1)
            total_correct += (predictions == test_labels).sum().item()
            total_samples += len(test_labels)

    final_accuracy = total_correct / total_samples

    elapsed = time.time() - start_time

    print(f"\n  Test Accuracy: {final_accuracy:.1%} ({total_correct}/{total_samples})")
    print(f"  Training Time: {elapsed/60:.1f} minutes")
    print(f"  Best at Gen: {trainer.generation}")
    print(f"  Model saved: {output_path}")

    print()
    if final_accuracy >= 0.90:
        print("  EXCELLENT! 90%+ accuracy - architecture works for real images!")
    elif final_accuracy >= 0.70:
        print("  SUCCESS! 70%+ accuracy - we have something real!")
    elif final_accuracy >= 0.50:
        print("  PARTIAL - Better than random, but needs improvement")
    else:
        print("  NEEDS WORK - Not much better than random guessing")

    return 0


if __name__ == "__main__":
    sys.exit(main())
