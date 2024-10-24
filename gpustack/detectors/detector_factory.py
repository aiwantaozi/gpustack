import logging
from typing import Dict
from gpustack.detectors.base import (
    GPUDetector,
    GPUDevicesInfo,
)
from gpustack.schemas.workers import SystemInfo, VendorEnum
from gpustack.detectors.fastfetch.fastfetch import Fastfetch
from gpustack.detectors.npu_smi.npu_smi import NPUSMI


logger = logging.getLogger(__name__)


class DetectorFactory:
    def __init__(
        self,
        default_gpu_detector: GPUDetector = None,
        gpu_detectors: Dict[VendorEnum, GPUDetector] = None,
    ):
        fastfetch = Fastfetch()
        self.system_info_detector = fastfetch
        self.gpu_vendor_detector = fastfetch
        self.default_gpu_detector = default_gpu_detector or fastfetch
        self.gpu_detectors = gpu_detectors

        if self.gpu_detectors is None:
            self.gpu_detectors = {
                VendorEnum.Huawei: NPUSMI(),
                VendorEnum.Apple: fastfetch,
                VendorEnum.NVIDIA: fastfetch,
                VendorEnum.MTHREADS: fastfetch,
            }

        self.system_info_detector_available = None
        self.gpu_vendor_detector_available = None

    def detect_gpus(self) -> GPUDevicesInfo:
        gpu_devices = []
        if len(self.gpu_detectors) == 0:
            gpu_devices.extend(self.default_gpu_detector.gather_gpu_info())
        else:
            if self.gpu_vendor_detector_available is None:
                self.gpu_vendor_detector_available = (
                    self.gpu_vendor_detector.is_available()
                )

            if not self.gpu_vendor_detector_available:
                raise Exception("GPU vendor detector is not available")

            vendors = self.gpu_vendor_detector.gather_gpu_vendor_info()
            filtered_vendors = [
                v
                for v in vendors
                if v.lower() in {m.lower() for m in VendorEnum.__members__.values()}
            ]
            for vendor in filtered_vendors:
                detector = self._get_gpu_detector_by_vendor(vendor)
                gpus = detector.gather_gpu_info()
                gpu_devices.extend(self._filter_gpu_devices(gpus, vendor))

        return gpu_devices

    def detect_system_info(self) -> SystemInfo:
        if self.system_info_detector_available is None:
            self.system_info_detector_available = (
                self.system_info_detector.is_available()
            )

        if not self.system_info_detector_available:
            raise Exception("System info detector is not available")

        return self.system_info_detector.gather_system_info()

    def _filter_gpu_devices(
        self, gpu_devices: GPUDevicesInfo, vendor: str
    ) -> GPUDevicesInfo:
        filtered_gpu_devices = []
        for device in gpu_devices:
            # Ignore the device without memory.
            if device.memory.total == 0:
                continue

            if device.vendor.lower() != vendor.lower():
                continue

            filtered_gpu_devices.append(device)

        return filtered_gpu_devices

    def _get_gpu_detector_by_vendor(self, vendor: str) -> GPUDetector:
        detector = next(
            (v for k, v in self.gpu_detectors.items() if k.lower() in vendor.lower()),
            self.default_gpu_detector,
        )
        self._check_detector_availability(detector, f"GPU detector for {vendor}")
        return detector

    def _check_detector_availability(self, detector, name: str):
        if not detector:
            raise Exception(f"{name} is empty")
        if not detector.is_available():
            raise Exception(f"{name} {detector.__class__.__name__} is not available")
