from pathlib import Path
import torch
from torch import nn


class DragMLP(nn.Module):
    """3 body-frame velocity components -> [Fx,Fy,Fz,Mx,My,Mz]."""
    def __init__(self, hidden=1024, normalize=False):
        super().__init__()
        self.hidden = hidden
        self.normalize = normalize
        self.net = nn.Sequential(nn.Linear(3, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 6))
        self.register_buffer("x_mean", torch.zeros(3))
        self.register_buffer("x_scale", torch.ones(3))
        self.register_buffer("y_mean", torch.zeros(6))
        self.register_buffer("y_scale", torch.ones(6))
        self.register_buffer("velocity_min", torch.zeros(3))
        self.register_buffer("velocity_max", torch.zeros(3))

    def forward(self, velocity):
        return self.net((velocity - self.x_mean) / self.x_scale) * self.y_scale + self.y_mean

    def normalized_prediction(self, velocity):
        return self.net((velocity - self.x_mean) / self.x_scale)


class SurrogateWrench(nn.Module):
    """Batched body-frame wrench. Inputs (...,3), outputs (...,6), SI units.

    Linear velocity must be relative to water. Moments must share the reference
    point used by the CFD dataset. This module adds no buoyancy or thruster force.
    """
    def __init__(self, translation, rotation):
        super().__init__()
        self.translation = translation
        self.rotation = rotation

    def forward(self, linear_velocity, angular_velocity):
        return self.translation(linear_velocity) + self.rotation(angular_velocity)


def load_branch(path, device="cpu"):
    checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
    if checkpoint["format_version"] != 1:
        raise ValueError("Unsupported checkpoint format")
    model = DragMLP(checkpoint["hidden"], checkpoint["normalize"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), checkpoint


def load_surrogate(translation_path, rotation_path, device="cpu"):
    translation, t = load_branch(translation_path, device)
    rotation, r = load_branch(rotation_path, device)
    if t["branch"] != "translation" or r["branch"] != "rotation":
        raise ValueError("Expected translation and rotation checkpoints, in that order")
    if t["frame"] != r["frame"] or t["moment_reference"] != r["moment_reference"]:
        raise ValueError("CFD frame and moment reference must match across branches")
    return SurrogateWrench(translation, rotation).to(device).eval()
