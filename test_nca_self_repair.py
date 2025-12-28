#!/usr/bin/env python3
"""
NCA Self-Repair Experiment

Tests whether NCA can regenerate functional neural network weights after damage.
Inspired by Mordvintsev's NCA image regeneration - if you damage the image and
run more NCA steps, it regenerates. Can the same happen with NN weights?

Experiment:
1. Train NCA to generate MNIST classifier
2. Generate working weights (baseline accuracy)
3. Apply damage (zero weights, noise, neuron deletion)
4. Run MORE NCA steps on damaged state
5. Measure if accuracy recovers

This tests if NCA learns true self-organizing dynamics vs one-shot generation.
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
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class DifferentiableNCA(nn.Module):
    """NCA that generates neural network weights with persistent state."""

    def __init__(self, level_sizes: list, channels: int = 8, device=None):
        super().__init__()
        self.level_sizes = level_sizes
        self.num_levels = len(level_sizes)
        self.channels = channels
        self.device = device or torch.device('cpu')
        self.neighbor_input = 27 * channels

        # Update networks for each level (smaller networks)
        self.update_nets = nn.ModuleList()
        for i in range(self.num_levels):
            input_size = self.neighbor_input
            if i < self.num_levels - 1:
                input_size += channels  # top-down signal

            self.update_nets.append(
                nn.Sequential(
                    nn.Linear(input_size, 32),
                    nn.ReLU(),
                    nn.Linear(32, channels),
                    nn.Tanh()
                ).to(self.device)
            )

    def _get_neighbor_features(self, state):
        batch, d, h, w, c = state.shape
        if d == 1 and h == 1 and w == 1:
            return state.reshape(batch, 1, 1, 1, c).repeat(1, 1, 1, 1, 27)

        padded = F.pad(
            state.permute(0, 4, 1, 2, 3),
            (1, 1, 1, 1, 1, 1),
            mode='replicate'
        ).permute(0, 2, 3, 4, 1)

        neighbors = []
        for di in range(3):
            for hi in range(3):
                for wi in range(3):
                    neighbors.append(padded[:, di:di+d, hi:hi+h, wi:wi+w, :])
        return torch.cat(neighbors, dim=-1)

    def _expand_to_lower(self, state):
        batch, d, h, w, c = state.shape
        expanded = state.unsqueeze(2).unsqueeze(4).unsqueeze(6)
        expanded = expanded.repeat(1, 1, 2, 1, 2, 1, 2, 1)
        return expanded.reshape(batch, d*2, h*2, w*2, c)

    def init_states(self, batch_size=1):
        """Initialize NCA states."""
        states = []
        for size in self.level_sizes:
            state = torch.randn(batch_size, size, size, size, self.channels,
                              device=self.device) * 0.1
            states.append(state)
        return states

    def step(self, states):
        """Run one NCA step, returns new states."""
        new_states = []
        for level in range(self.num_levels - 1, -1, -1):
            state = states[level]
            batch, d, h, w, c = state.shape

            neighbors = self._get_neighbor_features(state)

            if level < self.num_levels - 1:
                upper = new_states[0] if new_states else states[level + 1]
                top_down = self._expand_to_lower(upper)
                while top_down.shape[1] < state.shape[1]:
                    top_down = self._expand_to_lower(top_down)
                if top_down.shape[1] > state.shape[1]:
                    top_down = top_down[:, :d, :h, :w, :]
                update_input = torch.cat([neighbors, top_down], dim=-1)
            else:
                update_input = neighbors

            flat_input = update_input.reshape(-1, update_input.shape[-1])
            delta = self.update_nets[level](flat_input)
            delta = delta.reshape(batch, d, h, w, self.channels)
            new_state = state + delta * 0.1
            new_states.insert(0, new_state)

        return new_states

    def forward(self, states=None, steps=10, batch_size=1):
        """Run NCA for multiple steps."""
        if states is None:
            states = self.init_states(batch_size)

        for _ in range(steps):
            states = self.step(states)

        return states


class NCAWeightGenerator(nn.Module):
    """Full system: NCA + projections to generate classifier weights.

    Uses bottleneck projections to reduce memory usage.
    """

    def __init__(self, device):
        super().__init__()
        self.device = device

        # Smaller NCA
        level_sizes = [4, 2]  # Smaller levels
        channels = 8
        self.nca = DifferentiableNCA(level_sizes, channels=channels, device=device)

        # State sizes: 4^3*8=512, 2^3*8=64
        self.state_sizes = [(s ** 3) * channels for s in level_sizes]

        # Network architecture: 784 -> 64 -> 32 -> 10 (smaller network)
        self.hidden1, self.hidden2 = 64, 32
        self.input_size, self.output_size = 784, 10

        # Bottleneck size
        bottleneck = 64

        # Bottleneck projections (state -> bottleneck -> weights)
        self.encode_l0 = nn.Linear(self.state_sizes[0], bottleneck).to(device)
        self.encode_l1 = nn.Linear(self.state_sizes[1], bottleneck).to(device)

        # Weight projections from bottleneck
        self.proj_w1 = nn.Linear(bottleneck, self.input_size * self.hidden1).to(device)
        self.proj_b1 = nn.Linear(bottleneck, self.hidden1).to(device)
        self.proj_w2 = nn.Linear(bottleneck, self.hidden1 * self.hidden2).to(device)
        self.proj_b2 = nn.Linear(bottleneck, self.hidden2).to(device)
        self.proj_w3 = nn.Linear(bottleneck, self.hidden2 * self.output_size).to(device)
        self.proj_b3 = nn.Linear(bottleneck, self.output_size).to(device)

    def generate_weights_from_states(self, states):
        """Generate network weights from NCA states via bottleneck."""
        l0_flat = states[0].reshape(1, -1)
        l1_flat = states[1].reshape(1, -1)

        # Encode through bottleneck
        z0 = F.relu(self.encode_l0(l0_flat))
        z1 = F.relu(self.encode_l1(l1_flat))

        # Generate weights from bottleneck
        w1 = self.proj_w1(z0).reshape(self.hidden1, self.input_size)
        b1 = self.proj_b1(z0).reshape(self.hidden1)
        w2 = self.proj_w2(z1).reshape(self.hidden2, self.hidden1)
        b2 = self.proj_b2(z1).reshape(self.hidden2)
        w3 = self.proj_w3(z1).reshape(self.output_size, self.hidden2)
        b3 = self.proj_b3(z1).reshape(self.output_size)

        return (w1, b1, w2, b2, w3, b3)

    def forward_with_weights(self, x, weights):
        """Forward pass using generated weights."""
        w1, b1, w2, b2, w3, b3 = weights
        x = x.view(x.size(0), -1)
        x = F.relu(F.linear(x, w1, b1))
        x = F.relu(F.linear(x, w2, b2))
        x = F.linear(x, w3, b3)
        return x


def apply_damage(states, damage_type, damage_level=0.5):
    """Apply damage to NCA states.

    Args:
        states: List of NCA state tensors
        damage_type: 'zero' (zero out), 'noise' (add noise), 'region' (zero region)
        damage_level: Fraction of damage (0.0 to 1.0)

    Returns:
        Damaged states (new tensors, original unchanged)
    """
    damaged = []

    for state in states:
        state_copy = state.clone()

        if damage_type == 'zero':
            # Zero out random elements
            mask = torch.rand_like(state_copy) > damage_level
            state_copy = state_copy * mask.float()

        elif damage_type == 'noise':
            # Add Gaussian noise
            noise = torch.randn_like(state_copy) * damage_level * state_copy.std()
            state_copy = state_copy + noise

        elif damage_type == 'region':
            # Zero out a contiguous region (like cutting out a piece)
            _, d, h, w, _ = state_copy.shape
            # Calculate region size
            region_size = int(d * damage_level ** (1/3))
            if region_size > 0:
                start_d = np.random.randint(0, max(1, d - region_size))
                start_h = np.random.randint(0, max(1, h - region_size))
                start_w = np.random.randint(0, max(1, w - region_size))
                state_copy[:, start_d:start_d+region_size,
                          start_h:start_h+region_size,
                          start_w:start_w+region_size, :] = 0

        elif damage_type == 'half':
            # Zero out half the tensor (like cutting in half)
            _, d, h, w, _ = state_copy.shape
            state_copy[:, :d//2, :, :, :] = 0

        damaged.append(state_copy)

    return damaged


def evaluate_accuracy(model, states, test_loader, device):
    """Evaluate accuracy of network generated from states."""
    model.eval()
    weights = model.generate_weights_from_states(states)

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model.forward_with_weights(images, weights)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


def train_nca(model, train_loader, test_loader, device, epochs=5, nca_steps=10):
    """Train the NCA weight generator."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print("Training NCA weight generator...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            # Generate weights from NCA (fresh each batch)
            states = model.nca(steps=nca_steps)
            weights = model.generate_weights_from_states(states)

            # Forward pass
            outputs = model.forward_with_weights(images, weights)
            loss = criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if batch_idx % 100 == 0:
                print(f"  Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

            # Clear cache periodically
            if batch_idx % 50 == 0 and device.type == 'cuda':
                torch.cuda.empty_cache()

        # Evaluate
        model.eval()
        with torch.no_grad():
            states = model.nca(steps=nca_steps)
            test_acc = evaluate_accuracy(model, states, test_loader, device)
        print(f"Epoch {epoch+1}: Loss={total_loss/num_batches:.4f}, Test Acc={test_acc:.2f}%")

    return model


def run_self_repair_experiment(model, test_loader, device, nca_steps=10):
    """Run the self-repair experiment."""

    print("\n" + "="*70)
    print("  SELF-REPAIR EXPERIMENT")
    print("="*70)

    # Step 1: Generate baseline (healthy) weights
    print("\n1. Generating baseline network...")
    model.eval()

    with torch.no_grad():
        healthy_states = model.nca(steps=nca_steps)
        baseline_acc = evaluate_accuracy(model, healthy_states, test_loader, device)

    print(f"   Baseline accuracy: {baseline_acc:.2f}%")

    # Step 2: Test different damage types
    damage_types = ['zero', 'noise', 'region', 'half']
    damage_levels = [0.25, 0.5, 0.75]
    recovery_steps_list = [0, 5, 10, 20, 50]

    results = {}

    for damage_type in damage_types:
        print(f"\n2. Testing damage type: {damage_type.upper()}")
        print("-" * 60)

        for damage_level in damage_levels:
            if damage_type == 'half':
                damage_level = 0.5  # Half is always 50%

            key = f"{damage_type}_{int(damage_level*100)}"
            results[key] = {'damage_type': damage_type, 'damage_level': damage_level, 'recovery': {}}

            # Apply damage
            with torch.no_grad():
                damaged_states = apply_damage(healthy_states, damage_type, damage_level)

                # Measure accuracy immediately after damage
                damaged_acc = evaluate_accuracy(model, damaged_states, test_loader, device)
                results[key]['damaged_acc'] = damaged_acc

                print(f"\n   Damage {int(damage_level*100)}%: {baseline_acc:.2f}% -> {damaged_acc:.2f}%")

                # Try recovery with more NCA steps
                print(f"   Recovery attempts (more NCA steps on damaged state):")

                for recovery_steps in recovery_steps_list:
                    if recovery_steps == 0:
                        recovered_acc = damaged_acc
                    else:
                        # Run more NCA steps starting from damaged state
                        recovered_states = damaged_states
                        for _ in range(recovery_steps):
                            recovered_states = model.nca.step(recovered_states)
                        recovered_acc = evaluate_accuracy(model, recovered_states, test_loader, device)

                    results[key]['recovery'][recovery_steps] = recovered_acc
                    recovery_pct = (recovered_acc - damaged_acc) / (baseline_acc - damaged_acc + 1e-8) * 100

                    status = ""
                    if recovered_acc >= baseline_acc * 0.95:
                        status = " [RECOVERED]"
                    elif recovered_acc > damaged_acc + 5:
                        status = " [IMPROVING]"

                    print(f"     +{recovery_steps:2d} steps: {recovered_acc:.2f}% (recovery: {recovery_pct:+.1f}%){status}")

            if damage_type == 'half':
                break  # Only one level for half

    return results, baseline_acc


def main():
    parser = argparse.ArgumentParser(description="NCA Self-Repair Experiment")
    parser.add_argument('--epochs', type=int, default=5, help='Training epochs')
    parser.add_argument('--nca-steps', type=int, default=10, help='NCA steps for generation')
    parser.add_argument('--load', type=str, default=None, help='Load pretrained model')
    parser.add_argument('--save', type=str, default=None, help='Save trained model')
    args = parser.parse_args()

    print("="*70)
    print("  NCA SELF-REPAIR EXPERIMENT")
    print("  Can NCA regenerate functional weights after damage?")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Create or load model
    model = NCAWeightGenerator(device)

    if args.load and Path(args.load).exists():
        print(f"\nLoading model from {args.load}")
        model.load_state_dict(torch.load(args.load, map_location=device))
    else:
        print(f"\nTraining new model for {args.epochs} epochs...")
        model = train_nca(model, train_loader, test_loader, device,
                         epochs=args.epochs, nca_steps=args.nca_steps)

        if args.save:
            torch.save(model.state_dict(), args.save)
            print(f"Model saved to {args.save}")

    # Run self-repair experiment
    results, baseline = run_self_repair_experiment(
        model, test_loader, device, nca_steps=args.nca_steps
    )

    # Summary
    print("\n" + "="*70)
    print("  SUMMARY")
    print("="*70)
    print(f"\nBaseline accuracy: {baseline:.2f}%")

    print("\nRecovery Analysis:")
    print("-" * 60)

    full_recoveries = 0
    partial_recoveries = 0
    no_recovery = 0

    for key, data in results.items():
        damage_type = data['damage_type']
        damage_level = data['damage_level']
        damaged_acc = data['damaged_acc']
        best_recovery = max(data['recovery'].values())
        best_steps = max(data['recovery'], key=data['recovery'].get)

        recovery_pct = (best_recovery - damaged_acc) / (baseline - damaged_acc + 1e-8) * 100

        if best_recovery >= baseline * 0.95:
            status = "FULL RECOVERY"
            full_recoveries += 1
        elif best_recovery > damaged_acc + 5:
            status = "PARTIAL"
            partial_recoveries += 1
        else:
            status = "NO RECOVERY"
            no_recovery += 1

        print(f"{damage_type:6s} {int(damage_level*100):3d}%: "
              f"{damaged_acc:.1f}% -> {best_recovery:.1f}% @ {best_steps} steps [{status}]")

    print("-" * 60)
    print(f"Full recoveries: {full_recoveries}")
    print(f"Partial recoveries: {partial_recoveries}")
    print(f"No recovery: {no_recovery}")

    # Key insight
    print("\n" + "="*70)
    print("  KEY INSIGHT")
    print("="*70)

    if full_recoveries > partial_recoveries + no_recovery:
        print("\n  SUCCESS! NCA demonstrates self-repair capability.")
        print("  The self-organizing dynamics can regenerate functional weights.")
        print("  This is analogous to biological regeneration!")
    elif partial_recoveries > 0:
        print("\n  PARTIAL SUCCESS. NCA shows some regenerative ability.")
        print("  More NCA steps help recover from damage, but not fully.")
        print("  May need: longer training, different architecture, or repair-specific training.")
    else:
        print("\n  NCA does not self-repair in this configuration.")
        print("  The NCA may be a one-shot generator, not truly self-organizing.")
        print("  Next: Try training with damage augmentation to encourage repair.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
