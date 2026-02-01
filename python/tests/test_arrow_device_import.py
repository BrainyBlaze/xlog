import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")


def test_arrow_device_import_roundtrip():
    if not hasattr(pyxlog, "export_arrow_device") or not hasattr(pyxlog, "import_arrow_device"):
        pytest.skip("pyxlog built without the arrow-device-import feature")

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    x = torch.tensor([1, 2, 3], device="cuda", dtype=torch.int32)

    # Export a CUDA-resident Arrow C Device handle from a DLPack-producing column.
    dev = pyxlog.export_arrow_device([x], device=0, memory_mb=128)

    # Import back into DLPack columns and validate the bytes round-trip.
    tensors, names, num_rows = pyxlog.import_arrow_device(dev, device=0, memory_mb=128)
    assert num_rows == x.numel()
    assert len(names) == 1
    assert len(tensors) == 1

    from torch.utils.dlpack import from_dlpack

    out = from_dlpack(tensors[0])
    assert out.is_cuda
    assert out.dtype == x.dtype
    assert out.numel() == x.numel()
    assert out.cpu().tolist() == x.cpu().tolist()

