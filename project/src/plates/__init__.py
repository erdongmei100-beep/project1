"""Plate detection utilities."""

from .collector import PlateCollector, VehicleFilter
from .detector_yolo import PlateDetector

__all__ = ["PlateCollector", "PlateDetector", "VehicleFilter"]
