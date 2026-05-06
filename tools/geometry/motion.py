"""Motion analysis tool slot."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from peps.tools.base import BaseTool, LocalModelRequirement, ToolContext, ToolResult, ToolSpec


@dataclass(slots=True)
class MotionResult:
    mean_flow: list[float]
    avg_magnitude: float
    backend: str = "placeholder"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_message_content(self) -> str:
        return f"MotionResult: mean_flow={self.mean_flow}, avg_magnitude={self.avg_magnitude:.3f}"


class AnalyzeMotionTool(BaseTool):
    spec = ToolSpec(
        name="analyze_motion",
        description="Analyze motion between two sequential images, usually for camera movement cues.",
        input_schema={
            "type": "object",
            "required": ["image_source_1", "image_source_2"],
            "properties": {
                "image_source_1": {"type": "string"},
                "image_source_2": {"type": "string"},
            },
        },
        output_schema={"type": "object", "class": "MotionResult"},
        model_requirements=[
            LocalModelRequirement(
                name="OpenCV",
                required_for="Farneback optical flow local motion analysis",
                package="opencv-python",
                download_hint="No model weights required; install opencv-python for real optical flow.",
                required=False,
            )
        ],
        tags=["motion", "optical_flow"],
        is_placeholder=True,
    )

    async def _arun(
        self,
        *,
        context: ToolContext,
        image_source_1: Any,
        image_source_2: Any,
        **kwargs: Any,
    ) -> ToolResult:
        computed = _try_compute_farneback(image_source_1, image_source_2)
        if computed is not None:
            return self.success(computed)
        result = MotionResult(
            mean_flow=[0.0, 0.0],
            avg_magnitude=0.0,
            backend="placeholder",
            metadata={"note": "Install OpenCV/Pillow/NumPy for local optical flow."},
        )
        return self.success(result)


def _try_compute_farneback(image_source_1: Any, image_source_2: Any) -> MotionResult | None:
    try:
        import cv2
        import numpy as np
        from PIL import Image
    except Exception:
        return None

    try:
        img1 = _load_grayscale_array(image_source_1, Image, np)
        img2 = _load_grayscale_array(image_source_2, Image, np)
        if img1 is None or img2 is None or img1.shape != img2.shape:
            return None
        flow = cv2.calcOpticalFlowFarneback(
            prev=img1,
            next=img2,
            flow=None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        mean_flow = np.mean(flow, axis=(0, 1)).tolist()
        magnitudes = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        return MotionResult(
            mean_flow=[float(mean_flow[0]), float(mean_flow[1])],
            avg_magnitude=float(np.mean(magnitudes)),
            backend="opencv_farneback",
        )
    except Exception:
        return None


def _load_grayscale_array(source: Any, image_module: Any, np_module: Any) -> Any | None:
    if hasattr(source, "convert"):
        return np_module.array(source.convert("L"))
    if isinstance(source, str) and Path(source).exists():
        with image_module.open(source) as image:
            return np_module.array(image.convert("L"))
    return None

