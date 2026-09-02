import numpy as np
import torch

from gpuforecast.data import StandardScaler


def test_scaler_roundtrip():
    x = np.array([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]], dtype=np.float32)
    s = StandardScaler().fit(x)
    z = s.transform(x)
    restored = s.inverse_torch(torch.from_numpy(z)).numpy()
    np.testing.assert_allclose(restored, x, rtol=1e-5, atol=1e-5)
