import unittest
import torch
from coral_auv_surrogate.model import SurrogateWrench
from coral_auv_surrogate.runtime import WrenchRuntime


class Fixed(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.register_buffer('value', torch.tensor(value, dtype=torch.float32))

    def forward(self, velocity):
        return self.value.expand(*velocity.shape[:-1], 6)


class RuntimeTests(unittest.TestCase):
    def test_rotation_current_and_moment_reference(self):
        runtime = WrenchRuntime.__new__(WrenchRuntime)
        runtime.model = SurrogateWrench(Fixed([1,0,0,0,0,0]), Fixed([0,0,0,0,0,0]))
        r = torch.tensor([[[0.,-1,0],[1,0,0],[0,0,1]]])
        z = torch.zeros(1,3)
        force, torque = runtime.world_wrench(z,z,z,r,torch.tensor([0.,1,0]))
        torch.testing.assert_close(force, torch.tensor([[0.,1,0]]))
        torch.testing.assert_close(torque, torch.tensor([[0.,0,1]]))


if __name__ == '__main__':
    unittest.main()
