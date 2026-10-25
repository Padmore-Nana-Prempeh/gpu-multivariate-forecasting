"""
Phase 15: CUDA Extension Correctness Tests

Project: GPU-Accelerated Multivariate Forecasting

Operation:
    y = 2 * x

Tests:
    - FP16, FP32, FP64
    - Multiple tensor shapes
    - Non-multiple-of-256 sizes
    - Input preservation
    - Non-default CUDA stream
    - Invalid-input rejection

Requires:
    NVIDIA CUDA GPU
    Compiled trivial_cuda_ext extension

The original archived implementation does not handle
zero-element tensors safely; that case is excluded pending
a separately validated kernel fix.
"""

import torch


def main():
    print("=" * 65)
    print("PHASE 15: CUDA EXTENSION CORRECTNESS TESTS")
    print("=" * 65)

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA GPU unavailable. Run these tests in the "
            "validated NVIDIA GPU container."
        )

    import trivial_cuda_ext

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")

    torch.manual_seed(42)

    shapes = [
        (1,),
        (17,),
        (255,),
        (256,),
        (257,),
        (1000,),
        (32, 92),
        (256, 96, 7),
    ]

    dtypes = [
        torch.float16,
        torch.float32,
        torch.float64,
    ]

    passed = 0

    print("\n========== NUMERICAL CORRECTNESS ==========")

    for dtype in dtypes:
        for shape in shapes:

            x = torch.randn(
                shape,
                dtype=dtype,
                device="cuda",
            )

            original = x.clone()

            expected = x * 2
            actual = trivial_cuda_ext.scale2(x)

            torch.cuda.synchronize()

            if dtype == torch.float16:
                rtol, atol = 1e-3, 1e-3
            elif dtype == torch.float32:
                rtol, atol = 1e-6, 1e-6
            else:
                rtol, atol = 1e-12, 1e-12

            torch.testing.assert_close(
                actual,
                expected,
                rtol=rtol,
                atol=atol,
            )

            assert actual.shape == x.shape
            assert actual.dtype == x.dtype
            assert actual.device == x.device
            assert torch.equal(x, original)

            passed += 1

            print(
                f"PASS | dtype={str(dtype):15s} "
                f"| shape={str(shape)}"
            )

    print("\n========== CUDA STREAM CORRECTNESS ==========")

    stream = torch.cuda.Stream()

    with torch.cuda.stream(stream):
        x = torch.randn(
            4096,
            dtype=torch.float32,
            device="cuda",
        )

        actual = trivial_cuda_ext.scale2(x)
        expected = x * 2

    stream.synchronize()

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-6,
        atol=1e-6,
    )

    passed += 1
    print("PASS | Non-default CUDA stream")

    print("\n========== INPUT VALIDATION ==========")

    cpu_tensor = torch.randn(8)

    try:
        trivial_cuda_ext.scale2(cpu_tensor)
    except RuntimeError as exc:
        assert "CUDA" in str(exc)
        passed += 1
        print("PASS | CPU tensor rejected")
    else:
        raise AssertionError(
            "Expected CPU tensor rejection"
        )

    noncontiguous = torch.randn(
        4, 7, device="cuda"
    ).transpose(0, 1)

    assert not noncontiguous.is_contiguous()

    try:
        trivial_cuda_ext.scale2(noncontiguous)
    except RuntimeError as exc:
        assert "contiguous" in str(exc)
        passed += 1
        print("PASS | Noncontiguous tensor rejected")
    else:
        raise AssertionError(
            "Expected noncontiguous tensor rejection"
        )

    print("\n" + "=" * 65)
    print(f"ALL {passed} TEST CASES PASSED")
    print("=" * 65)


if __name__ == "__main__":
    main()
