"""
Hierarchical Neural Cellular Automata - PyTorch Implementation

Configurable N-level architecture inspired by biological hierarchy:
- Level 0 (bottom): Highest resolution - "subatomic/physics"
- Level N (top): Lowest resolution - "organism/decision"

Default 6-level configuration (2x scaling):
  L0: 32³ = 32,768 cells (Subatomic)
  L1: 16³ = 4,096 cells  (Atomic)
  L2: 8³ = 512 cells     (Molecular)
  L3: 4³ = 64 cells      (Organelle)
  L4: 2³ = 8 cells       (Cellular)
  L5: 1³ = 1 cell        (Organism - single decision neuron)

Weight layout per level (1852 weights):
- LOCAL_W: 27 * 4 * 16 = 1728 (neighbor input -> hidden)
- LOCAL_B: 16 (hidden biases)
- OUT_W: 16 * 4 = 64 (hidden -> output)
- OUT_B: 4 (output biases)
- UP_W: 4 * 4 = 16 (upward summary projection)
- UP_B: 4 (upward summary bias)
- DOWN_W: 4 * 4 = 16 (downward modulation projection)
- DOWN_B: 4 (downward modulation bias)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


# Constants matching Unity shader architecture
CHANNELS = 4
HIDDEN_SIZE = 16
NEIGHBORHOOD = 27
WEIGHTS_PER_LEVEL = 1852

# Weight offsets within each level
LOCAL_W_SIZE = NEIGHBORHOOD * CHANNELS * HIDDEN_SIZE  # 1728
LOCAL_B_SIZE = HIDDEN_SIZE  # 16
OUT_W_SIZE = HIDDEN_SIZE * CHANNELS  # 64
OUT_B_SIZE = CHANNELS  # 4
UP_W_SIZE = CHANNELS * CHANNELS  # 16
UP_B_SIZE = CHANNELS  # 4
DOWN_W_SIZE = CHANNELS * CHANNELS  # 16
DOWN_B_SIZE = CHANNELS  # 4

LOCAL_W_OFFSET = 0
LOCAL_B_OFFSET = LOCAL_W_SIZE
OUT_W_OFFSET = LOCAL_B_OFFSET + LOCAL_B_SIZE
OUT_B_OFFSET = OUT_W_OFFSET + OUT_W_SIZE
UP_W_OFFSET = OUT_B_OFFSET + OUT_B_SIZE
UP_B_OFFSET = UP_W_OFFSET + UP_W_SIZE
DOWN_W_OFFSET = UP_B_OFFSET + UP_B_SIZE
DOWN_B_OFFSET = DOWN_W_OFFSET + DOWN_W_SIZE


@dataclass
class HierarchyConfig:
    """Configuration for hierarchical NCA."""
    level_sizes: List[int]  # Size of each level (e.g., [32, 16, 8, 4, 2, 1])
    input_size: int  # Input pattern size (e.g., 32 for 32x32)

    @property
    def num_levels(self) -> int:
        return len(self.level_sizes)

    @property
    def total_weights(self) -> int:
        return self.num_levels * WEIGHTS_PER_LEVEL

    @property
    def total_cells(self) -> int:
        return sum(s ** 3 for s in self.level_sizes)


# Preset configurations
CONFIG_4_LEVEL = HierarchyConfig(
    level_sizes=[16, 8, 4, 2],
    input_size=16
)

CONFIG_6_LEVEL = HierarchyConfig(
    level_sizes=[32, 16, 8, 4, 2, 1],
    input_size=32
)

CONFIG_8_LEVEL = HierarchyConfig(
    level_sizes=[64, 32, 16, 8, 4, 2, 1, 1],  # Two 1³ levels at top
    input_size=64
)

# Default to 6 levels
DEFAULT_CONFIG = CONFIG_6_LEVEL


class HierarchicalNCA:
    """
    Configurable N-level Hierarchical NCA.

    Mirrors biological hierarchy from subatomic to organism level.
    Each level processes information and passes summaries up / receives modulation down.
    """

    def __init__(self, config: HierarchyConfig = None, device: torch.device = None):
        self.config = config or DEFAULT_CONFIG
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Weights as a flat array
        self.weights = np.zeros(self.config.total_weights, dtype=np.float32)

    @property
    def num_levels(self) -> int:
        return self.config.num_levels

    @property
    def level_sizes(self) -> List[int]:
        return self.config.level_sizes

    @property
    def total_weights(self) -> int:
        return self.config.total_weights

    def set_weights(self, weights: np.ndarray):
        """Set weights from a flat numpy array."""
        expected = self.config.total_weights
        assert len(weights) == expected, f"Expected {expected} weights, got {len(weights)}"
        self.weights = weights.astype(np.float32)

    def get_weights(self) -> np.ndarray:
        """Get weights as flat numpy array."""
        return self.weights.copy()

    def get_level_weights(self, level: int) -> np.ndarray:
        """Get weights for a specific level."""
        start = level * WEIGHTS_PER_LEVEL
        return self.weights[start:start + WEIGHTS_PER_LEVEL]

    def get_level_weight_ranges(self) -> List[Tuple[int, int]]:
        """Get (start, end) indices for each level's weights."""
        return [
            (i * WEIGHTS_PER_LEVEL, (i + 1) * WEIGHTS_PER_LEVEL)
            for i in range(self.num_levels)
        ]

    def initialize_state(self, batch_size: int) -> List[torch.Tensor]:
        """Initialize state tensors for all levels."""
        states = []
        for size in self.level_sizes:
            state = torch.zeros(batch_size, size, size, size, CHANNELS, device=self.device)
            states.append(state)
        return states

    def set_input(self, states: List[torch.Tensor], input_pattern: torch.Tensor):
        """
        Set input pattern into L0 (bottom level).
        input_pattern: (batch, input_size, input_size) binary/float pattern
        """
        batch_size = input_pattern.shape[0]
        input_h, input_w = input_pattern.shape[1], input_pattern.shape[2]
        level_size = self.level_sizes[0]

        # Handle size mismatch via interpolation if needed
        if input_h != level_size or input_w != level_size:
            # Resize input to match L0 size
            input_resized = F.interpolate(
                input_pattern.unsqueeze(1),  # Add channel dim
                size=(level_size, level_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(1)
        else:
            input_resized = input_pattern

        # Expand 2D to 3D by repeating along Z
        expanded = input_resized.unsqueeze(-1).expand(-1, -1, -1, level_size)
        states[0][:, :, :, :, 0] = expanded

    def _get_neighbor_input(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get 27-neighbor input for each cell (including self).
        state: (batch, size, size, size, channels)
        returns: (batch, size, size, size, 27*channels)
        """
        batch_size, size = state.shape[0], state.shape[1]

        if size == 1:
            # Special case: 1x1x1 grid, just repeat self 27 times
            # Use repeat, not expand - expand only works on singleton dims
            return state.repeat(1, 1, 1, 1, NEIGHBORHOOD)  # (batch, 1, 1, 1, 108)

        # Pad with wrap-around
        x = state.permute(0, 4, 1, 2, 3)  # (batch, channels, x, y, z)
        padded = F.pad(x, (1, 1, 1, 1, 1, 1), mode='circular')

        # Extract 27 neighbors for each cell
        neighbors = []
        for dx in range(3):
            for dy in range(3):
                for dz in range(3):
                    neighbor = padded[:, :,
                                     dx:dx+size,
                                     dy:dy+size,
                                     dz:dz+size]
                    neighbors.append(neighbor)

        # Stack and reshape
        stacked = torch.stack(neighbors, dim=0)
        result = stacked.permute(1, 3, 4, 5, 0, 2).reshape(batch_size, size, size, size, -1)
        return result

    def _apply_level_network(self, neighbor_input: torch.Tensor, child_summary: torch.Tensor,
                             modulation: torch.Tensor, level: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply the neural network for one level.
        Returns: (new_state, upward_summary)
        """
        batch_size = neighbor_input.shape[0]
        size = neighbor_input.shape[1]
        n_cells = size ** 3

        # Get level weights
        w = self.get_level_weights(level)
        w_t = torch.from_numpy(w).to(self.device)

        # Extract weight matrices
        local_w = w_t[LOCAL_W_OFFSET:LOCAL_W_OFFSET + LOCAL_W_SIZE].reshape(NEIGHBORHOOD * CHANNELS, HIDDEN_SIZE)
        local_b = w_t[LOCAL_B_OFFSET:LOCAL_B_OFFSET + LOCAL_B_SIZE]
        out_w = w_t[OUT_W_OFFSET:OUT_W_OFFSET + OUT_W_SIZE].reshape(HIDDEN_SIZE, CHANNELS)
        out_b = w_t[OUT_B_OFFSET:OUT_B_OFFSET + OUT_B_SIZE]
        up_w = w_t[UP_W_OFFSET:UP_W_OFFSET + UP_W_SIZE].reshape(CHANNELS, CHANNELS)
        up_b = w_t[UP_B_OFFSET:UP_B_OFFSET + UP_B_SIZE]
        down_w = w_t[DOWN_W_OFFSET:DOWN_W_OFFSET + DOWN_W_SIZE].reshape(CHANNELS, CHANNELS)
        down_b = w_t[DOWN_B_OFFSET:DOWN_B_OFFSET + DOWN_B_SIZE]

        # Flatten spatial dims
        flat_neighbor = neighbor_input.reshape(batch_size * n_cells, -1)

        # Handle child summary
        if child_summary is not None:
            flat_child = child_summary.reshape(batch_size * n_cells, -1)
            # Add child influence to neighbor input
            child_expanded = flat_child.repeat(1, NEIGHBORHOOD)
            flat_neighbor = flat_neighbor + child_expanded * 0.1  # Scale factor

        # Handle modulation
        mod_scale = None
        if modulation is not None:
            flat_mod = modulation.reshape(batch_size * n_cells, -1)
            mod_scale = torch.sigmoid(flat_mod @ down_w + down_b)

        # Hidden layer
        hidden = F.relu(flat_neighbor @ local_w + local_b)

        # Apply modulation
        if mod_scale is not None:
            hidden = hidden * mod_scale.repeat(1, HIDDEN_SIZE // CHANNELS)

        # Output layer
        output = torch.sigmoid(hidden @ out_w + out_b)

        # Upward summary
        summary = torch.tanh(output @ up_w + up_b)

        # Reshape back
        new_state = output.reshape(batch_size, size, size, size, CHANNELS)
        summary_out = summary.reshape(batch_size, size, size, size, CHANNELS)

        return new_state, summary_out

    def _compute_child_summary(self, child_state: torch.Tensor, parent_size: int) -> torch.Tensor:
        """
        Summarize child level for parent level.
        Handles non-2x scaling via adaptive pooling.
        """
        if child_state is None:
            return None

        batch_size = child_state.shape[0]
        child_size = child_state.shape[1]

        if child_size == parent_size:
            # Same size - just pass through
            return child_state

        if child_size == parent_size * 2:
            # Clean 2x reduction - average 2x2x2 blocks
            x = child_state.reshape(batch_size, parent_size, 2, parent_size, 2, parent_size, 2, CHANNELS)
            return x.mean(dim=(2, 4, 6))

        # General case: use adaptive average pooling
        # Reshape for pooling: (batch, channels, x, y, z)
        x = child_state.permute(0, 4, 1, 2, 3)
        pooled = F.adaptive_avg_pool3d(x, (parent_size, parent_size, parent_size))
        return pooled.permute(0, 2, 3, 4, 1)

    def _expand_modulation(self, parent_state: torch.Tensor, child_size: int) -> torch.Tensor:
        """
        Expand parent state to child size.
        Handles non-2x scaling via interpolation.
        """
        if parent_state is None:
            return None

        parent_size = parent_state.shape[1]

        if parent_size == child_size:
            return parent_state

        if parent_size * 2 == child_size:
            # Clean 2x expansion
            x = parent_state
            x = x.repeat_interleave(2, dim=1)
            x = x.repeat_interleave(2, dim=2)
            x = x.repeat_interleave(2, dim=3)
            return x

        # General case: use interpolation
        x = parent_state.permute(0, 4, 1, 2, 3)  # (batch, channels, x, y, z)
        expanded = F.interpolate(x, size=(child_size, child_size, child_size), mode='trilinear', align_corners=False)
        return expanded.permute(0, 2, 3, 4, 1)

    def step(self, states: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Perform one step of the hierarchical CA.

        Order:
        1. Compute child summaries going UP
        2. Process top level first (no parent modulation)
        3. Process remaining levels top-down with modulation
        """
        batch_size = states[0].shape[0]
        new_states = [None] * self.num_levels

        # Phase 1: Get neighbor inputs for all levels
        neighbor_inputs = [self._get_neighbor_input(s) for s in states]

        # Phase 2: Compute child summaries going UP
        child_summaries = [None] * self.num_levels
        for level in range(1, self.num_levels):
            child_summaries[level] = self._compute_child_summary(
                states[level - 1],
                self.level_sizes[level]
            )

        # Phase 3: Process top level (no modulation)
        top_level = self.num_levels - 1
        new_states[top_level], _ = self._apply_level_network(
            neighbor_inputs[top_level],
            child_summaries[top_level],
            None,
            level=top_level
        )

        # Phase 4: Process remaining levels top-down
        for level in range(top_level - 1, -1, -1):
            modulation = self._expand_modulation(
                new_states[level + 1],
                self.level_sizes[level]
            )
            new_states[level], _ = self._apply_level_network(
                neighbor_inputs[level],
                child_summaries[level],
                modulation,
                level=level
            )

        return new_states

    def forward(self, input_pattern: torch.Tensor, steps: int = 20) -> torch.Tensor:
        """
        Run the hierarchical CA for multiple steps.
        input_pattern: (batch, height, width) binary/float pattern
        returns: scalar output (batch,) - average of top level channel 0
        """
        batch_size = input_pattern.shape[0]
        states = self.initialize_state(batch_size)
        self.set_input(states, input_pattern)

        for _ in range(steps):
            states = self.step(states)

        # Output: average of top level state channel 0
        top_state = states[self.num_levels - 1]
        output = top_state[:, :, :, :, 0].mean(dim=(1, 2, 3))
        return output

    def __call__(self, input_pattern: torch.Tensor, steps: int = 20) -> torch.Tensor:
        return self.forward(input_pattern, steps)


def get_weight_count(config: HierarchyConfig = None) -> int:
    """Get total weight count for a configuration."""
    config = config or DEFAULT_CONFIG
    return config.total_weights


def get_level_weight_ranges(config: HierarchyConfig = None) -> List[Tuple[int, int]]:
    """Get (start, end) indices for each level's weights."""
    config = config or DEFAULT_CONFIG
    return [
        (i * WEIGHTS_PER_LEVEL, (i + 1) * WEIGHTS_PER_LEVEL)
        for i in range(config.num_levels)
    ]


def print_config_info(config: HierarchyConfig):
    """Print detailed information about a hierarchy configuration."""
    print(f"\n{'='*60}")
    print(f"Hierarchical NCA Configuration: {config.num_levels} Levels")
    print(f"{'='*60}")

    level_names = [
        "Subatomic", "Atomic", "Molecular", "Macromolecular",
        "Organelle", "Cellular", "Circuit", "Tissue",
        "Organ", "System", "Organism"
    ]

    total_cells = 0
    for i, size in enumerate(config.level_sizes):
        cells = size ** 3
        total_cells += cells
        name = level_names[i] if i < len(level_names) else f"Level {i}"
        print(f"  L{i} ({name:14s}): {size:3d}³ = {cells:>8,} cells")

    print(f"  {'─'*45}")
    print(f"  Total cells: {total_cells:,}")
    print(f"  Weights per level: {WEIGHTS_PER_LEVEL}")
    print(f"  Total weights: {config.total_weights:,}")
    print(f"  Input size: {config.input_size}x{config.input_size}")
    print()


if __name__ == "__main__":
    print("Available configurations:")
    print("\n4-Level (Original):")
    print_config_info(CONFIG_4_LEVEL)

    print("\n6-Level (Recommended):")
    print_config_info(CONFIG_6_LEVEL)

    print("\n8-Level (Extended):")
    print_config_info(CONFIG_8_LEVEL)

    # Quick test with 6-level
    print("\nTesting 6-level configuration...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = HierarchicalNCA(CONFIG_6_LEVEL, device)
    model.set_weights(np.random.randn(model.total_weights).astype(np.float32) * 0.1)

    # Test forward pass
    batch_size = 4
    input_pattern = (torch.rand(batch_size, 32, 32, device=device) > 0.5).float()

    print(f"Running forward pass with {batch_size} samples...")
    import time
    start = time.time()
    output = model(input_pattern, steps=10)
    elapsed = time.time() - start

    print(f"Input shape: {input_pattern.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output: {output.detach().cpu().numpy()}")
    print(f"Time: {elapsed:.3f}s")
