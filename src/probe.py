"""
Probe Architectures for MIC

Contribution 1: Prove that Stable Audio VAE latent space
encodes interpretable musical concepts (pitch).

If LinearProbe works well -> pitch info is linearly decodable
If CNNProbe works better -> temporal context helps
"""

import torch
import torch.nn as nn


class LinearProbe(nn.Module):
    """
    Simple linear probe for pitch class prediction.

    If this works well, it demonstrates that pitch information
    is linearly separable in the VAE latent space.
    """

    def __init__(
        self,
        in_channels: int = 64,
        out_classes: int = 12,
        use_layer_norm: bool = True
    ):
        super().__init__()

        self.use_layer_norm = use_layer_norm

        if use_layer_norm:
            self.norm = nn.LayerNorm(in_channels)

        self.proj = nn.Linear(in_channels, out_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: VAE latent (B, C, T)

        Returns:
            logits: (B, T, out_classes)
        """
        # (B, C, T) -> (B, T, C)
        z = z.permute(0, 2, 1)

        if self.use_layer_norm:
            z = self.norm(z)

        logits = self.proj(z)
        return logits

    def predict(self, z: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Get binary predictions."""
        logits = self.forward(z)
        probs = torch.sigmoid(logits)
        return (probs > threshold).float()


class CNNProbe(nn.Module):
    """
    Small CNN probe that can capture temporal context.

    Slightly more powerful than LinearProbe, useful for
    comparison in ablation study.
    """

    def __init__(
        self,
        in_channels: int = 64,
        hidden_channels: int = 128,
        out_classes: int = 12,
        kernel_size: int = 5,
        num_layers: int = 2
    ):
        super().__init__()
        if hidden_channels % 8 != 0:
            raise ValueError(f"hidden_channels must be divisible by 8 (GroupNorm groups=8), got {hidden_channels}")

        layers = []

        # Input layer
        layers.append(nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=kernel_size // 2))
        layers.append(nn.GroupNorm(8, hidden_channels))
        layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2))
            layers.append(nn.GroupNorm(8, hidden_channels))
            layers.append(nn.ReLU())

        self.features = nn.Sequential(*layers)

        # Output layer
        self.output = nn.Conv1d(hidden_channels, out_classes, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: VAE latent (B, C, T)

        Returns:
            logits: (B, T, out_classes)
        """
        h = self.features(z)  # (B, hidden, T)
        logits = self.output(h)  # (B, out_classes, T)
        logits = logits.permute(0, 2, 1)  # (B, T, out_classes)
        return logits

    def predict(self, z: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Get binary predictions."""
        logits = self.forward(z)
        probs = torch.sigmoid(logits)
        return (probs > threshold).float()


def create_probe(
    probe_type: str = "cnn",
    in_channels: int = 64,
    hidden_channels: int = 128,
    out_classes: int = 12,
    **kwargs
) -> nn.Module:
    """Factory function to create probe."""
    if probe_type.lower() == "linear":
        return LinearProbe(in_channels=in_channels, out_classes=out_classes)
    elif probe_type.lower() == "cnn":
        return CNNProbe(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_classes=out_classes,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown probe type: {probe_type}")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick test
    print("Testing probe architectures...")

    batch_size = 4
    channels = 64
    time_steps = 215

    dummy_latent = torch.randn(batch_size, channels, time_steps)

    # Test LinearProbe
    linear_probe = LinearProbe(in_channels=64, out_classes=12)
    output = linear_probe(dummy_latent)
    print(f"LinearProbe: {count_parameters(linear_probe):,} params")
    print(f"  Input: {dummy_latent.shape}")
    print(f"  Output: {output.shape}")

    # Test CNNProbe
    cnn_probe = CNNProbe(in_channels=64, hidden_channels=128, out_classes=12)
    output = cnn_probe(dummy_latent)
    print(f"CNNProbe: {count_parameters(cnn_probe):,} params")
    print(f"  Input: {dummy_latent.shape}")
    print(f"  Output: {output.shape}")

    print("\nAll tests passed!")
