"""Geometry tool placeholders and deterministic helpers."""

from peps.tools.geometry.detection import BoundingBox2D, DetectionResult, DetectionTool
from peps.tools.geometry.motion import AnalyzeMotionTool, MotionResult
from peps.tools.geometry.ocr import OCRResult, OCRText, OCRTool
from peps.tools.geometry.pose import ObjectPoseResult, PredictObjectPoseTool
from peps.tools.geometry.projection import ProjectionResult, ProjectBoxTo3DPointsTool
from peps.tools.geometry.reconstruction import CameraPose, ReconstructionResult, ReconstructionTool
from peps.tools.geometry.scale import EstimateScaleTool, MetricScaleResult

__all__ = [
    "AnalyzeMotionTool",
    "BoundingBox2D",
    "CameraPose",
    "DetectionResult",
    "DetectionTool",
    "EstimateScaleTool",
    "MetricScaleResult",
    "MotionResult",
    "OCRResult",
    "OCRText",
    "OCRTool",
    "ObjectPoseResult",
    "PredictObjectPoseTool",
    "ProjectionResult",
    "ProjectBoxTo3DPointsTool",
    "ReconstructionResult",
    "ReconstructionTool",
]

