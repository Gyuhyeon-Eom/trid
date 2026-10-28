import unittest
import torch
from trid.model import Attention,Config,rotary


class RotaryCacheTests(unittest.TestCase):
    def test_cached_rotations_match_reference_and_gradients(self):
        for device in ['cpu']+(['mps'] if torch.backends.mps.is_available() else []):
            cfg=Config(dim=32,heads=4,layers=1,context_length=48,rope_theta=25000)
            attention=Attention(cfg).to(device)
            self.assertNotIn('rope_cos',attention.state_dict())
            for offset,length in [(0,21),(19,1),(29,19)]:
                x=torch.randn(2,4,length,8,device=device,requires_grad=True)
                y=x.detach().clone().requires_grad_()
                reference=rotary(x,offset,cfg.rope_theta)
                cached=rotary(y,offset,cfg.rope_theta,(attention.rope_cos,attention.rope_sin))
                # CPU and MPS sin/cos kernels can differ slightly in float32.
                torch.testing.assert_close(cached,reference,atol=2e-6,rtol=2e-6)
                reference.square().sum().backward();cached.square().sum().backward()
                torch.testing.assert_close(y.grad,x.grad,atol=4e-6,rtol=3e-6)
