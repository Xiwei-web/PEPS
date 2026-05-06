"""Semantic detection tool slot."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from peps.tools.base import BaseTool, LocalModelRequirement, ToolContext, ToolResult, ToolSpec


@dataclass(slots=True)
class BoundingBox2D:
    x1: float
    y1: float
    x2: float
    y2: float
    label: str = "object"
    score: float = 1.0
    view_id: str | None = None

    def as_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass(slots=True)
class DetectionResult:
    boxes: list[BoundingBox2D] = field(default_factory=list)
    detector_type: str = "placeholder"
    prompt: str | list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def labels(self) -> list[str]:
        return [box.label for box in self.boxes]

    def to_message_content(self) -> str:
        if not self.boxes:
            return "DetectionResult: no objects detected by the current placeholder backend."
        return f"DetectionResult: detected {len(self.boxes)} object(s): {self.labels}"


class DetectionTool(BaseTool):
    spec = ToolSpec(
        name="detect",
        description="Detect objects in an image from text prompts.",
        input_schema={
            "type": "object",
            "required": ["image_source", "prompt"],
            "properties": {
                "image_source": {"type": "string"},
                "prompt": {"type": ["string", "array"]},
                "precomputed_boxes": {"type": "array"},
            },
        },
        output_schema={"type": "object", "class": "DetectionResult"},
        model_requirements=[
            LocalModelRequirement(
                name="GroundingDINO",
                required_for="open-vocabulary object detection",
                local_path_env="PEPS_GROUNDING_DINO_MODEL_DIR",
                checkpoint_env="PEPS_GROUNDING_DINO_CHECKPOINT",
                package="groundingdino",
                download_hint="Download GroundingDINO config/checkpoint locally or use an OpenAI VLM detector backend later.",
                required=False,
            )
        ],
        tags=["perception", "detection"],
        is_placeholder=True,
    )

    async def _arun(
        self,
        *,
        context: ToolContext,
        image_source: Any,
        prompt: str | list[str],
        precomputed_boxes: list[Any] | None = None,
        allow_full_image_fallback: bool = False,
        view_id: str | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        boxes = [_coerce_box(row, view_id=view_id) for row in precomputed_boxes or []]
        if not boxes and allow_full_image_fallback:
            size = _try_image_size(image_source)
            if size is not None:
                width, height = size
                label = prompt[0] if isinstance(prompt, list) and prompt else str(prompt)
                boxes = [BoundingBox2D(0.0, 0.0, float(width), float(height), label=label, score=0.1, view_id=view_id)]
        result = DetectionResult(
            boxes=boxes,
            detector_type="precomputed_or_placeholder",
            prompt=prompt,
            metadata={
                "image_source": str(image_source),
                "note": "Configure GroundingDINO or VLM detector backend for real detections.",
            },
        )
        return self.success(result)


def _coerce_box(row: Any, *, view_id: str | None = None) -> BoundingBox2D:
    if isinstance(row, BoundingBox2D):
        return row
    if isinstance(row, dict):
        coords = row.get("box") or row.get("bbox") or [row["x1"], row["y1"], row["x2"], row["y2"]]
        label = row.get("label", "object")
        score = row.get("score", 1.0)
        return BoundingBox2D(*[float(item) for item in coords], label=label, score=float(score), view_id=row.get("view_id", view_id))
    coords = [float(item) for item in row]
    return BoundingBox2D(*coords[:4], view_id=view_id)


def _try_image_size(source: Any) -> tuple[int, int] | None:
    if hasattr(source, "size") and isinstance(source.size, tuple):
        return source.size
    if isinstance(source, str) and Path(source).exists():
        try:
            from PIL import Image

            with Image.open(source) as image:
                return image.size
        except Exception:
            return None
    return None

