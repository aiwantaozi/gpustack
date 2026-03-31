import importlib


def test_images_command_registers_evaluation_runner(monkeypatch):
    captured = []

    def fake_append_images(*images):
        captured.extend(images)

    monkeypatch.setattr("gpustack_runtime.cmds.append_images", fake_append_images)

    import gpustack.cmd.images as images_module

    importlib.reload(images_module)

    assert any("gpustack/evaluation-runner:" in image for image in captured)
