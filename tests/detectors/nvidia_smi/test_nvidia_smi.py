import os

from gpustack.detectors.nvidia_smi.nvidia_smi import NvidiaSMI
from gpustack.schemas.workers import MemoryInfo, SystemInfo, VendorEnum
from tests.detectors.rocm_smi.test_rocm_smi import gpu_device


def test_decode_gpu_devices():
    files = [
        "dgx-spark.txt",
    ]

    expected_outputs = [
        {
            "gpus": [
                gpu_device(
                    "",
                    "NVIDIA GB10",
                    0,
                    VendorEnum.NVIDIA.value,
                    2,
                    1,
                    0,
                    0.0,
                    42.0,
                    "",
                    True,
                ),
            ],
        },
    ]

    for i, file in enumerate(files):
        info_output = command_output(file)
        nvidia_smi = NvidiaSMI(gather_system_info_func=mock_gather_system_info)
        devices = nvidia_smi.decode_gpu_devices(info_output)

        assert expected_outputs[i].get("gpus") == devices


def mock_gather_system_info() -> SystemInfo:
    return SystemInfo(
        memory=MemoryInfo(
            total=2097152,  # 2 MB
            used=1048576,  # 1 MB
            utilization_rate=50.0,
        )
    )


def command_output(file: str) -> str:
    info_output = ""

    current_dir = os.path.dirname(__file__)

    info_file = os.path.join(current_dir, "data", f"{file}")
    with open(info_file, 'r') as f:
        info_output = f.read()

    return info_output
