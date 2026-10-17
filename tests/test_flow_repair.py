import pytest
import torch

from experiments.flow_repair_20261002 import NAME, focal_elements, training_adapter


def test_unit_invariance_and_finite_gradient():
    raw_p = torch.tensor([2., 12., 160.], dtype=torch.float64)
    raw_y = torch.tensor([1., 10., 100.], dtype=torch.float64)
    expected = None
    for unit in (0.01, 1., 100.):
        mean, std = 80.*unit, 150.*unit
        p = ((raw_p*unit-mean)/std).requires_grad_()
        y = (raw_y*unit-mean)/std
        value = focal_elements(p, y, dict(std=std))
        grad, = torch.autograd.grad(value.sum(), p)
        assert torch.isfinite(grad).all() and (grad > 0).all()
        if expected is None:
            expected = value
        else:
            torch.testing.assert_close(value, expected)


def test_actual_loss_path_masks_targets_and_restores_original():
    from experiments import v4_runner as v4
    original, name = v4.focal_elements, v4.FOCAL_NAME
    p = torch.tensor([1., 2., 999.]).reshape(1, 1, 3, 1).requires_grad_()
    batch = dict(target=torch.zeros_like(p), loss_mask=torch.tensor([[[True, True, False]]]),
                 lifecycle_context=torch.eye(4)[:3].reshape(1, 1, 3, 4))
    output = dict(pred=p, forecast_loss_mode='role_balanced_mix', forecast_role_balanced_beta=0.)
    with training_adapter():
        result = v4.compute_losses(output, batch, NAME, dict(std=150.))
        torch.testing.assert_close(result['optimize'], (p.flatten()[:2]*torch.tanh(p.flatten()[:2])).mean())
        result['optimize'].backward()
        assert p.grad.flatten()[2] == 0
        with pytest.raises(v4.RunnerRefused):
            v4.compute_losses(p, batch, NAME, dict(std=150.))
    assert v4.focal_elements is original and v4.FOCAL_NAME == name


def test_refiner_zero_scale_is_identity_with_trained_parameters():
    from model.ALFNet.ALFNet import GraphOutputRefiner
    m = GraphOutputRefiner(1, 8, 1, 0., {})
    with torch.no_grad():
        m.output_proj.weight.fill_(0.8)
        m.output_proj.bias.fill_(0.5)
    x = torch.randn(2, 12, 3, 1, requires_grad=True)
    role = torch.eye(4)[torch.tensor([[1, 2, 3], [0, 2, 3]])]
    y = m(x, torch.ones(3, 3), role)
    torch.testing.assert_close(y, x, rtol=0, atol=0)
    y.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x), rtol=0, atol=0)


def test_invalid_scaler_rejected():
    for std in (0., -1., float('nan')):
        with pytest.raises(ValueError):
            focal_elements(torch.zeros(1), torch.zeros(1), dict(std=std))
