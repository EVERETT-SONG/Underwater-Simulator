"""Tensor-only inference boundary for a future IsaacSim physics callback."""
import torch
from .model import load_surrogate


class WrenchRuntime:
    def __init__(self, translation_checkpoint, rotation_checkpoint, device="cpu"):
        self.model = load_surrogate(translation_checkpoint, rotation_checkpoint, device)

    @torch.inference_mode()
    def body_wrench(self, relative_linear_velocity_body, angular_velocity_body):
        """Inputs (...,3), output (...,6), about the CFD moment reference."""
        return self.model(relative_linear_velocity_body, angular_velocity_body)

    @torch.inference_mode()
    def world_wrench(self, velocity_world, omega_world, water_world, body_to_world,
                     reference_to_com_body):
        """Velocities evaluated at CFD origin; return world force/COM torque.

        reference_to_com_body = COM - CFD_reference in body coordinates, metres.
        All tensors must have compatible batch shapes, device and float32 dtype.
        body_to_world is a rotation matrix (...,3,3), not a quaternion.
        """
        def rotate(matrix, vector):
            return (matrix @ vector.unsqueeze(-1)).squeeze(-1)
        world_to_body = body_to_world.transpose(-1, -2)
        wrench = self.body_wrench(rotate(world_to_body, velocity_world-water_world),
                                 rotate(world_to_body, omega_world))
        force, moment = wrench[..., :3], wrench[..., 3:]
        moment_com = moment - torch.linalg.cross(reference_to_com_body.expand_as(force), force)
        return rotate(body_to_world, force), rotate(body_to_world, moment_com)
