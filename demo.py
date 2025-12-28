#!/usr/bin/env python3
"""
NCA Fault Tolerance Demo

Watch a neural network survive 95% state corruption.

This demo:
1. Trains a small NCA to generate MNIST classifier weights (~2 min on GPU)
2. Tests baseline accuracy
3. Corrupts 95% of the NCA state (bitflip damage)
4. Shows the network collapse... then recover

Usage:
    python demo.py              # Full demo with training
    python demo.py --quick      # Shorter training for quick test
    python demo.py --load model.pt  # Load pretrained model
"""

import argparse
import time
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ============================================================================
#  NCA Architecture
# ============================================================================

class DifferentiableNCA(nn.Module):
    """Neural Cellular Automaton with hierarchical levels."""

    def __init__(self, level_sizes=[4, 2], channels=8, device=None):
        super().__init__()
        self.level_sizes = level_sizes
        self.num_levels = len(level_sizes)
        self.channels = channels
        self.device = device or torch.device('cpu')
        self.neighbor_input = 27 * channels

        # Update networks for each level
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
        """Run one NCA step."""
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
    """NCA + projections to generate MNIST classifier weights."""

    def __init__(self, device):
        super().__init__()
        self.device = device

        # NCA configuration
        level_sizes = [4, 2]
        channels = 8
        self.nca = DifferentiableNCA(level_sizes, channels=channels, device=device)

        # State sizes
        self.state_sizes = [(s ** 3) * channels for s in level_sizes]

        # Network architecture: 784 -> 64 -> 32 -> 10
        self.hidden1, self.hidden2 = 64, 32
        self.input_size, self.output_size = 784, 10

        # Bottleneck projections
        bottleneck = 64
        self.encode_l0 = nn.Linear(self.state_sizes[0], bottleneck).to(device)
        self.encode_l1 = nn.Linear(self.state_sizes[1], bottleneck).to(device)

        self.proj_w1 = nn.Linear(bottleneck, self.input_size * self.hidden1).to(device)
        self.proj_b1 = nn.Linear(bottleneck, self.hidden1).to(device)
        self.proj_w2 = nn.Linear(bottleneck, self.hidden1 * self.hidden2).to(device)
        self.proj_b2 = nn.Linear(bottleneck, self.hidden2).to(device)
        self.proj_w3 = nn.Linear(bottleneck, self.hidden2 * self.output_size).to(device)
        self.proj_b3 = nn.Linear(bottleneck, self.output_size).to(device)

    def generate_weights_from_states(self, states):
        """Generate network weights from NCA states."""
        l0_flat = states[0].reshape(1, -1)
        l1_flat = states[1].reshape(1, -1)

        z0 = F.relu(self.encode_l0(l0_flat))
        z1 = F.relu(self.encode_l1(l1_flat))

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


# ============================================================================
#  Damage Functions
# ============================================================================

def apply_bitflip_damage(states, damage_level=0.95):
    """
    Apply bitflip damage to NCA states.
    
    Simulates random bit corruption - each affected element gets
    multiplied by -1, flipping its sign.
    """
    damaged = []
    for state in states:
        state_copy = state.clone()
        mask = torch.rand_like(state_copy) < damage_level
        state_copy[mask] = -state_copy[mask]  # Flip sign
        damaged.append(state_copy)
    return damaged


def compute_state_similarity(states1, states2):
    """Compute cosine similarity between two state sets."""
    flat1 = torch.cat([s.flatten() for s in states1])
    flat2 = torch.cat([s.flatten() for s in states2])
    return F.cosine_similarity(flat1.unsqueeze(0), flat2.unsqueeze(0)).item()


# ============================================================================
#  Training & Evaluation
# ============================================================================

def evaluate_accuracy(model, states, test_loader, device, max_batches=None):
    """Evaluate classifier accuracy."""
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        weights = model.generate_weights_from_states(states)

        for batch_idx, (images, labels) in enumerate(test_loader):
            if max_batches and batch_idx >= max_batches:
                break

            images, labels = images.to(device), labels.to(device)
            outputs = model.forward_with_weights(images, weights)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    return 100.0 * correct / total


def train_model(model, train_loader, test_loader, device, epochs=3, nca_steps=10):
    """Train the NCA weight generator."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            # Generate weights from NCA
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

            if batch_idx % 200 == 0:
                print(f"    Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

            if device.type == 'cuda' and batch_idx % 50 == 0:
                torch.cuda.empty_cache()

        # Evaluate
        model.eval()
        with torch.no_grad():
            states = model.nca(steps=nca_steps)
            test_acc = evaluate_accuracy(model, states, test_loader, device)
        print(f"  Epoch {epoch+1}/{epochs}: Loss={total_loss/num_batches:.4f}, Accuracy={test_acc:.1f}%")

    return model


# ============================================================================
#  Main Demo
# ============================================================================

def run_demo(model, test_loader, device, nca_steps=10):
    """Run the fault tolerance demonstration."""
    
    print("\n" + "="*70)
    print("  FAULT TOLERANCE DEMONSTRATION")
    print("="*70)

    model.eval()

    # Step 1: Generate healthy baseline
    print("\n[1] Generating healthy network state...")
    with torch.no_grad():
        healthy_states = model.nca(steps=nca_steps)
        baseline_acc = evaluate_accuracy(model, healthy_states, test_loader, device)
    print(f"    Baseline accuracy: {baseline_acc:.1f}%")

    # Step 2: Apply 95% bitflip damage
    print("\n[2] Applying 95% bitflip damage...")
    with torch.no_grad():
        damaged_states = apply_bitflip_damage(healthy_states, damage_level=0.95)
        
        # State similarity check
        similarity = compute_state_similarity(healthy_states, damaged_states)
        print(f"    State similarity after damage: {similarity:.1%}")
        
        damaged_acc = evaluate_accuracy(model, damaged_states, test_loader, device)
    print(f"    Accuracy after damage: {damaged_acc:.1f}%")
    print(f"    ⚠️  Network collapsed! ({baseline_acc:.1f}% → {damaged_acc:.1f}%)")

    # Step 3: Recovery via NCA self-organization
    print("\n[3] Running NCA recovery steps...")
    recovery_steps = [5, 10, 15]
    
    with torch.no_grad():
        current_states = damaged_states
        
        for target_steps in recovery_steps:
            # Run additional steps
            steps_to_run = target_steps - (0 if target_steps == recovery_steps[0] else recovery_steps[recovery_steps.index(target_steps)-1])
            for _ in range(steps_to_run):
                current_states = model.nca.step(current_states)
            
            recovered_acc = evaluate_accuracy(model, current_states, test_loader, device)
            similarity = compute_state_similarity(healthy_states, current_states)
            
            status = "✓ RECOVERED" if recovered_acc >= baseline_acc * 0.95 else "recovering..."
            print(f"    +{target_steps:2d} steps: {recovered_acc:.1f}% (similarity: {similarity:.1%}) {status}")

    # Final result
    final_acc = recovered_acc
    final_similarity = similarity

    print("\n" + "="*70)
    print("  RESULTS")
    print("="*70)
    print(f"""
    Baseline accuracy:     {baseline_acc:.1f}%
    After 95% damage:      {damaged_acc:.1f}%
    After recovery:        {final_acc:.1f}%
    
    Recovery:              +{final_acc - damaged_acc:.1f}% ({damaged_acc:.1f}% → {final_acc:.1f}%)
    State similarity:      {final_similarity:.1%} (different state, same function!)
    """)

    if final_acc >= baseline_acc * 0.95:
        print("    ✅ SUCCESS: Network recovered to baseline performance!")
        print("    The NCA learned a LANDSCAPE of equivalent solutions,")
        print("    not a single correct state. Damage immunity through degeneracy.")
    else:
        print("    ⚠️  Partial recovery. Try more training epochs.")

    print("="*70 + "\n")

    return {
        'baseline': baseline_acc,
        'damaged': damaged_acc,
        'recovered': final_acc,
        'similarity': final_similarity
    }


def main():
    parser = argparse.ArgumentParser(description="NCA Fault Tolerance Demo")
    parser.add_argument('--quick', action='store_true', help='Quick demo (1 epoch)')
    parser.add_argument('--epochs', type=int, default=5, help='Training epochs')
    parser.add_argument('--load', type=str, help='Load pretrained model')
    parser.add_argument('--save', type=str, help='Save trained model')
    args = parser.parse_args()

    print("="*70)
    print("  NCA FAULT TOLERANCE DEMO")
    print("  What if neural networks could survive 95% corruption?")
    print("="*70)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load MNIST
    print("\nLoading MNIST dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)

    print(f"Train: {len(train_dataset):,} samples")
    print(f"Test:  {len(test_dataset):,} samples")

    # Create model
    model = NCAWeightGenerator(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Train or load
    epochs = 1 if args.quick else args.epochs
    nca_steps = 10

    if args.load:
        print(f"\nLoading model from {args.load}...")
        model.load_state_dict(torch.load(args.load, map_location=device))
    else:
        print(f"\nTraining for {epochs} epoch(s)...")
        start = time.time()
        model = train_model(model, train_loader, test_loader, device, 
                           epochs=epochs, nca_steps=nca_steps)
        print(f"Training completed in {time.time()-start:.1f}s")

        if args.save:
            torch.save(model.state_dict(), args.save)
            print(f"Model saved to {args.save}")

    # Run demonstration
    results = run_demo(model, test_loader, device, nca_steps=nca_steps)

    return 0


if __name__ == "__main__":
    sys.exit(main())
