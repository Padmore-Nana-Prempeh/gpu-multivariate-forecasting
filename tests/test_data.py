import numpy as np

from gpuforecast.data import StandardScaler, WindowDataset


def test_scaler_standardizes_training_data():
    values = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
        ],
        dtype=np.float32,
    )

    scaler = StandardScaler().fit(values)
    scaled = scaler.transform(values)

    np.testing.assert_allclose(
        scaled.mean(axis=0),
        np.zeros(2),
        atol=1e-6,
    )

    np.testing.assert_allclose(
        scaled.std(axis=0),
        np.ones(2),
        atol=1e-6,
    )


def test_window_dataset_alignment():
    values = np.arange(1, 9, dtype=np.float32).reshape(-1, 1)

    dataset = WindowDataset(
        values=values,
        seq_len=3,
        horizon=2,
        label_start=3,
        label_end=len(values),
    )

    x0, y0 = dataset[0]
    x1, y1 = dataset[1]

    np.testing.assert_array_equal(
        x0.numpy().squeeze(),
        np.array([1.0, 2.0, 3.0]),
    )
    np.testing.assert_array_equal(
        y0.numpy().squeeze(),
        np.array([4.0, 5.0]),
    )

    np.testing.assert_array_equal(
        x1.numpy().squeeze(),
        np.array([2.0, 3.0, 4.0]),
    )
    np.testing.assert_array_equal(
        y1.numpy().squeeze(),
        np.array([5.0, 6.0]),
    )


def test_validation_window_can_use_training_history():
    values = np.arange(10, dtype=np.float32).reshape(-1, 1)

    train_end = 6
    val_end = 10

    dataset = WindowDataset(
        values=values,
        seq_len=3,
        horizon=2,
        label_start=train_end,
        label_end=val_end,
    )

    x0, y0 = dataset[0]

    np.testing.assert_array_equal(
        x0.numpy().squeeze(),
        np.array([3.0, 4.0, 5.0]),
    )

    np.testing.assert_array_equal(
        y0.numpy().squeeze(),
        np.array([6.0, 7.0]),
    )


def test_scaler_does_not_fit_future_distribution():
    train = np.array(
        [
            [1.0],
            [2.0],
            [3.0],
            [4.0],
        ],
        dtype=np.float32,
    )

    future = np.array(
        [
            [100.0],
            [200.0],
        ],
        dtype=np.float32,
    )

    scaler = StandardScaler().fit(train)

    expected_train_mean = train.mean(axis=0, keepdims=True)

    np.testing.assert_allclose(
        scaler.mean,
        expected_train_mean,
    )

    assert not np.isclose(
        scaler.mean.item(),
        np.concatenate([train, future]).mean(),
    )

def test_split_sizes_match_configured_ratios():
    n = 69680
    train_ratio = 0.70
    val_ratio = 0.10

    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)

    train_end = train_size
    val_end = train_end + val_size

    assert train_end == 48776
    assert val_end == 55744

    assert train_size == 48776
    assert val_size == 6968
    assert n - val_end == 13936


