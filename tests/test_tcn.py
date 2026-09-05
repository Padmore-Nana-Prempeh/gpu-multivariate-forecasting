import torch

from gpuforecast.models import TCNBlock


def test_tcn_block_is_causal():
    torch.manual_seed(42)

    block = TCNBlock(
        in_ch=1,
        out_ch=4,
        kernel=3,
        dilation=2,
        dropout=0.0,
    )
    block.eval()

    x = torch.randn(1, 1, 20)

    t = 10

    with torch.no_grad():
        original = block(x)

        changed = x.clone()

        # Drastically alter observations AFTER time t.
        changed[:, :, t + 1 :] += 1000.0

        modified = block(changed)

    torch.testing.assert_close(
        original[:, :, t],
        modified[:, :, t],
    )
