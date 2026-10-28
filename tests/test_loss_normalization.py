import unittest
import torch
from torch import nn
from trid.training import loss_for, validation_loss


class FixedLogits(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.tensor([[2., 0.], [0., 2.]]))

    def forward(self, x, return_hidden=False):
        out = self.logits[x]
        return out, None, out


class LossNormalizationTests(unittest.TestCase):
    def test_equal_example_weight_despite_answer_length(self):
        model = FixedLogits()
        x = torch.tensor([[0, 0, 0, 0], [1, 1, 1, 1]])
        y = torch.tensor([[0, -100, -100, -100], [0, 0, 0, 0]])
        short = loss_for(model, x[:1], y[:1])
        long = loss_for(model, x[1:], y[1:])
        balanced = loss_for(model, x, y, loss_normalization='example')
        torch.testing.assert_close(balanced, (short + long) / 2)
        torch.testing.assert_close(loss_for(model, x, y), (short + 4 * long) / 5)
        balanced.backward()
        expected = model.logits.grad.clone()
        model.zero_grad()
        ((loss_for(model, x[:1], y[:1]) + loss_for(model, x[1:], y[1:])) / 2).backward()
        torch.testing.assert_close(model.logits.grad, expected)

    def test_validation_invariant_to_uneven_batches_and_padding(self):
        model = FixedLogits()
        examples = [([0], [0]), ([1, 1, 1], [0, 0, 0]), ([0, 0], [-100, 0])]
        for mode in ('token', 'example'):
            scores = [validation_loss(model, examples, 0, 'cpu', size, loss_normalization=mode)
                      for size in (1, 2, 3)]
            for score in scores[1:]:
                self.assertAlmostEqual(scores[0], score, places=6)

    def test_missing_answer_targets_rejected(self):
        with self.assertRaisesRegex(ValueError, 'answer target'):
            loss_for(FixedLogits(), torch.tensor([[0]]), torch.tensor([[-100]]),
                     loss_normalization='example')
