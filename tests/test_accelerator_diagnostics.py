from genesis_ai import accelerator_diagnostics


def test_collect_diagnostics_cpu(monkeypatch):
    monkeypatch.setattr(
        accelerator_diagnostics.torch.cuda,
        "is_available",
        lambda: False,
    )
    monkeypatch.setattr(
        accelerator_diagnostics.platform,
        "processor",
        lambda: "Test CPU",
    )

    info = accelerator_diagnostics.collect_diagnostics()

    assert info["device"] == "cpu"
    assert info["device_name"] == "Test CPU"
    assert info["cuda_available"] is False
    assert info["cuda_version"] is None


def test_format_diagnostics_cpu():
    info = {
        "platform": "Linux",
        "platform_release": "test",
        "python_version": "3.11.0",
        "pytorch_version": "2.0.0",
        "device": "cpu",
        "device_name": "Test CPU",
        "cuda_available": False,
        "cuda_version": None,
    }

    output = accelerator_diagnostics.format_diagnostics(info)

    assert "Platform: Linux test" in output
    assert "Python: 3.11.0" in output
    assert "PyTorch: 2.0.0" in output
    assert "Device: cpu" in output
    assert "CUDA available: False" in output
    assert "CPU-only runtime detected" in output


def test_collect_diagnostics_cuda(monkeypatch):
    monkeypatch.setattr(
        accelerator_diagnostics.torch.cuda,
        "is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        accelerator_diagnostics.torch.cuda,
        "get_device_name",
        lambda index: "Test GPU",
    )
    monkeypatch.setattr(
        accelerator_diagnostics.torch.version,
        "cuda",
        "12.1",
    )

    info = accelerator_diagnostics.collect_diagnostics()

    assert info["device"] == "cuda"
    assert info["device_name"] == "Test GPU"
    assert info["cuda_available"] is True
    assert info["cuda_version"] == "12.1"


def test_output_does_not_include_environment_variables(monkeypatch):
    monkeypatch.setenv("KAGGLE_KEY", "super-secret-value")
    monkeypatch.setenv("HF_TOKEN", "another-secret-value")

    output = accelerator_diagnostics.format_diagnostics(
        {
            "platform": "Linux",
            "platform_release": "test",
            "python_version": "3.11.0",
            "pytorch_version": "2.0.0",
            "device": "cpu",
            "device_name": "CPU",
            "cuda_available": False,
            "cuda_version": None,
        }
    )

    assert "super-secret-value" not in output
    assert "another-secret-value" not in output