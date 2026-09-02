import torch

from gpuforecast.models import RNNForecaster, TCNForecaster, TransformerForecaster


def test_model_shapes():
    x = torch.randn(4, 96, 7)
    models = [
        RNNForecaster("lstm", 7, 24, 32, 1, 0.0),
        RNNForecaster("gru", 7, 24, 32, 1, 0.0),
        TCNForecaster(7, 24, [16, 32], 0.0),
        TransformerForecaster(7, 24, 32, 1, 4, 0.0),
    ]
    for model in models:
        assert model(x).shape == (4, 24, 7)
