import json
import os
from gpustack.scheduler.calculator import modelResoruceClaim
from gpustack.schemas.workers import (
    GPUDeviceInfo,
    MemoryInfo,
    SystemReserved,
    Worker,
    WorkerStatus,
)

# The data structure is compatible with the gguf-parser-go (v0.9.2), data values are designed according to test needs


def llama3_8b_partial_offload_estimate_claim():
    '''
    gguf-parser --ol-model llama3:8b \
    --gpu-layers=-1 --gpu-layers-step=1 --ctx-size 8192 \
    --skip-tokenizer --skip-architecture --skip-metadata --json >> llama3_8b_partial_offload_estimate_claim.json
    '''
    return load_model_estimate_claim_from_file(
        "llama3_8b_partial_offload_estimate_claim.json"
    )


def llama3_8b_full_offload_estimate_claim():
    '''
    gguf-parser --ol-model llama3:8b \
    --gpu-layers=-1 --ctx-size 8192 \
    --skip-tokenizer --skip-architecture --skip-metadata --json >> llama3_8b_full_offload_estimate_claim.json
    '''
    return load_model_estimate_claim_from_file(
        "llama3_8b_full_offload_estimate_claim.json"
    )


def llama3_8b_disable_offload_estimate_claim():
    '''
    gguf-parser --ol-model llama3:8b \
    --gpu-layers=0 --ctx-size=8192  \
    --skip-tokenizer --skip-architecture --skip-metadata --json >> llama3_8b_disable_offload_estimate_claim.json
    '''
    return load_model_estimate_claim_from_file(
        "llama3_8b_disable_offload_estimate_claim.json"
    )


def llama3_70b_partial_offload_estimate_claim():
    '''
    gguf-parser --ol-model llama3:70b \
    --gpu-layers=-1 --gpu-layers-step=1 --ctx-size 8192 \
    --skip-tokenizer --skip-architecture --skip-metadata --json >> llama3_70b_partial_offload_estimate_claim.json
    '''
    return load_model_estimate_claim_from_file(
        "llama3_70b_partial_offload_estimate_claim.json"
    )


def llama3_70b_full_offload_estimate_claim():
    '''
    gguf-parser --ol-model llama3:70b \
    --gpu-layers=-1 --ctx-size 8192 \
    --skip-tokenizer --skip-architecture --skip-metadata --json >> llama3_70b_full_offload_estimate_claim.json
    '''
    return load_model_estimate_claim_from_file(
        "llama3_70b_full_offload_estimate_claim.json"
    )


def llama3_70b_disable_offload_estimate_claim():
    '''
    gguf-parser --ol-model llama3:70b \
    --gpu-layers=0 --ctx-size 8192 \
    --skip-tokenizer --skip-architecture --skip-metadata --json >> llama3_70b_disable_offload_estimate_claim.json
    '''
    return load_model_estimate_claim_from_file(
        "llama3_70b_disable_offload_estimate_claim.json"
    )


def llama3_70b_full_offload_split_2_4080_estimate_claim():
    '''
    gguf-parser --ol-model llama3:70b \
    --gpu-layers=-1 --ctx-size 8192 \
    --tensor-split=17171480576,17171480576 \
    --skip-tokenizer --skip-architecture --skip-metadata --json >> llama3_70b_full_offload_split_2_4080_estimate_claim.json
    '''
    return load_model_estimate_claim_from_file(
        "llama3_70b_full_offload_split_2_4080_estimate_claim.json"
    )


def llama3_70b_full_offload_split_2_4080_estimate_claim_2():
    '''
    gguf-parser --ol-model llama3:70b \
    --gpu-layers=-1 --ctx-size 8192 \
    --tensor-split=17171480576,16647192576 \
    --skip-tokenizer --skip-architecture --skip-metadata --json >> llama3_70b_full_offload_split_2_4090_estimate_claim_2.json
    or
    gguf-parser --ol-model llama3:70b \
    --gpu-layers=-1 --ctx-size 8192 \
    --tensor-split=17171480576,16542334976 \
    --skip-tokenizer --skip-architecture --skip-metadata --json >> llama3_70b_full_offload_split_2_4090_estimate_claim_2.json
    '''
    return load_model_estimate_claim_from_file(
        "llama3_70b_full_offload_split_2_4090_estimate_claim_2.json"
    )


def llama3_70b_full_offload_split_2_4080_estimate_claim_3():
    '''
    gguf-parser --ol-model llama3:70b \
    --gpu-layers=-1 --ctx-size 8192 \
    --tensor-split=16647192576,16542334976 \
    --skip-tokenizer --skip-architecture --skip-metadata --json >> llama3_70b_full_offload_split_2_4090_estimate_claim_3.json
    '''
    return load_model_estimate_claim_from_file(
        "llama3_70b_full_offload_split_2_4090_estimate_claim_3.json"
    )


def llama3_70b_full_offload_split_3_4080_estimate_claim():
    '''
    gguf-parser --ol-model llama3:70b \
    --gpu-layers=-1 --ctx-size 8192 \
    --tensor-split=17171480576,17171480576,16647192576 \
    --skip-tokenizer --skip-architecture --skip-metadata --json >> llama3_70b_full_offload_split_3_4080_estimate_claim.json
    or
    gguf-parser --ol-model llama3:70b \
    --gpu-layers=-1 --ctx-size 8192 \
    --tensor-split=17171480576,17171480576,16542334976 \
    --skip-tokenizer --skip-architecture --skip-metadata --json >> llama3_70b_full_offload_split_3_4080_estimate_claim.json
    '''
    return load_model_estimate_claim_from_file(
        "llama3_70b_full_offload_split_3_4080_estimate_claim.json"
    )


def llama3_70b_full_offload_split_3_4080_estimate_claim_2():
    '''
    gguf-parser --ol-model llama3:70b \
    --gpu-layers=-1 --ctx-size 8192 \
    --tensor-split=17171480576,16647192576,16542334976 \
    --skip-tokenizer --skip-architecture --skip-metadata --json >> llama3_70b_full_offload_split_3_4080_estimate_claim_2.json
    '''
    return load_model_estimate_claim_from_file(
        "llama3_70b_full_offload_split_3_4080_estimate_claim_2.json"
    )


def llama3_70b_full_offload_split_3_4080_estimate_claim_3():
    '''
    gguf-parser --ol-model llama3:70b \
    --gpu-layers=-1 --ctx-size 8192 \
    --tensor-split=17171480576,17171480576,17171480576 \
    --skip-tokenizer --skip-architecture --skip-metadata --json >> llama3_70b_full_offload_split_3_4080_estimate_claim_3.json
    '''
    return load_model_estimate_claim_from_file(
        "llama3_70b_full_offload_split_3_4080_estimate_claim_3.json"
    )


def llama3_70b_full_offload_split_4_4080_estimate_claim():
    '''
    gguf-parser --ol-model llama3:70b \
    --gpu-layers=-1 --ctx-size 8192 \
    --tensor-split=17171480576,17171480576,17171480576,17171480576 \
    --skip-tokenizer --skip-architecture --skip-metadata --json >> llama3_70b_full_offload_split_4_4080_estimate_claim.json
    '''
    return load_model_estimate_claim_from_file(
        "llama3_70b_full_offload_split_4_4080_estimate_claim.json"
    )


def llama3_70b_full_offload_split_2_4090_estimate_claim():
    '''
    gguf-parser --ol-model llama3:70b \
    --gpu-layers=-1 --ctx-size 8192 \
    --tensor-split=26015170560,26015170560 \
    --skip-tokenizer --skip-architecture --skip-metadata --json >> llama3_70b_full_offload_split_2_4090_estimate_claim.json
    '''
    return load_model_estimate_claim_from_file(
        "llama3_70b_full_offload_split_2_4090_estimate_claim.json"
    )


def llama3_70b_partial_offload_split_2_4080_estimate_claim():
    '''
    gguf-parser --ol-model llama3:70b \
    --gpu-layers=-1 --gpu-layers-step=1 --ctx-size 8192 \
    --tensor-split=17171480576,17171480576 \
    --skip-tokenizer --skip-architecture --skip-metadata --json >> llama3_70b_partial_offload_split_2_4080_estimate_claim.json
    '''
    return load_model_estimate_claim_from_file(
        "llama3_70b_partial_offload_split_2_4080_estimate_claim.json"
    )


def llama3_70b_partial_offload_split_2_4080_4090_estimate_claim():
    '''
    gguf-parser --ol-model llama3:70b \
    --gpu-layers=-1 --gpu-layers-step=1 --ctx-size 8192 \
    --tensor-split=26015170560,17171480576 \
    --skip-tokenizer --skip-architecture --skip-metadata --json >> llama3_70b_partial_offload_split_2_4080_4090_estimate_claim.json
    '''
    return load_model_estimate_claim_from_file(
        "llama3_70b_partial_offload_split_2_4080_4090_estimate_claim.json"
    )


def llama3_70b_partial_offload_split_3_4080_4090_estimate_claim():
    '''
    gguf-parser --ol-model llama3:70b \
    --gpu-layers=-1 --gpu-layers-step=1 --ctx-size 8192 \
    --tensor-split=17171480576,17171480576,26015170560 --rpc=host:50020 \
    --skip-tokenizer --skip-architecture --skip-metadata --json >> llama3_70b_partial_offload_split_3_4080_4090_estimate_claim.json
    '''
    return load_model_estimate_claim_from_file(
        "llama3_70b_partial_offload_split_3_4080_4090_estimate_claim.json"
    )


def llama3_70b_partial_offload_split_3_4080_4090_estimate_claim_2():
    '''
    gguf-parser --ol-model llama3:70b \
    --gpu-layers=-1 --gpu-layers-step=1 --ctx-size 8192 \
    --tensor-split=26015170560,17171480576,17171480576 --rpc=host:50020,host:50021 \
    --skip-tokenizer --skip-architecture --skip-metadata --json >> llama3_70b_partial_offload_split_3_4080_4090_estimate_claim_2.json
    '''
    return load_model_estimate_claim_from_file(
        "llama3_70b_partial_offload_split_3_4080_4090_estimate_claim_2.json"
    )


def worker_macos_metal(reserved=False):
    return load_worker_from_file("worker_macos_metal.json", reserved=reserved)


def worker_linux_nvidia_1_4090_gpu(reserved=False):
    return load_worker_from_file(
        "worker_linux_nvidia_1_4090_gpu.json", reserved=reserved
    )


def worker_linux_nvidia_2_4080_gpu(reserved=False):
    return load_worker_from_file(
        "worker_linux_nvidia_2_4080_gpu.json", reserved=reserved
    )


def worker_linux_nvidia_2_4090_gpu(reserved=False):
    return load_worker_from_file(
        "worker_linux_nvidia_2_4090_gpu.json", reserved=reserved
    )


def worker_linux_nvidia_4_4080_gpu(reserved=False):
    return load_worker_from_file(
        "worker_linux_nvidia_4_4080_gpu.json", reserved=reserved
    )


def worker_linux_cpu_1(reserved=False):
    return load_worker_from_file("worker_linux_cpu_1.json", reserved=reserved)


def worker_linux_cpu_2(reserved=False):
    return load_worker_from_file("worker_linux_cpu_2.json", reserved=reserved)


def load_worker_from_file(file_name, reserved=False) -> Worker:
    dir = os.path.dirname(__file__)
    file_path = os.path.join(dir, file_name)
    with open(file_path, 'r') as file:
        worker_dict = json.loads(file.read())
        status_dict = worker_dict.get("status")
        memory = status_dict.get("memory")
        gpu_devices = status_dict.get("gpu_devices")

        status = WorkerStatus(**status_dict)
        status.memory = MemoryInfo(**memory)

        if gpu_devices:
            status.gpu_devices = [GPUDeviceInfo(**device) for device in gpu_devices]

        worker = Worker(**worker_dict)
        worker.status = status
        worker.system_reserved = SystemReserved(memory=0, gpu_memory=0)

        if reserved:
            system_reserved_dict = worker_dict.get("system_reserved")
            system_reserved = SystemReserved(**system_reserved_dict)
            worker.system_reserved = system_reserved
    return worker


def load_model_estimate_claim_from_file(file_name) -> modelResoruceClaim:
    dir = os.path.dirname(__file__)
    file_path = os.path.join(dir, file_name)
    with open(file_path, 'r') as file:
        model_estimate_claim = modelResoruceClaim.from_json(file.read())
    return model_estimate_claim
