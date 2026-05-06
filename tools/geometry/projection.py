"""Projection from 2D boxes into approximate 3D points."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from peps.tools.base import BaseTool, ToolContext, ToolResult, ToolSpec
from peps.tools.geometry.detection import BoundingBox2D
from peps.tools.geometry.geometry_ops import bbox_center
from peps.tools.geometry.reconstruction import ReconstructionResult


@dataclass(slots=True)
class ProjectionResult:
    points_3d: list[list[float]]
    points_confidence: list[float]
    selected_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_message_content(self) -> str:
        return f"ProjectionResult: {len(self.points_3d)} point(s), selected_index={self.selected_index}"


class ProjectBoxTo3DPointsTool(BaseTool):
    spec = ToolSpec(
        name="project_box_to_3d_points",
        description="Project a 2D bounding box into 3D scene points using a reconstruction.",
        input_schema={
            "type": "object",
            "required": ["reconstruction", "box"],
            "properties": {
                "reconstruction": {"type": "object"},
                "box": {"type": "array"},
                "selected_index": {"type": "integer"},
            },
        },
        output_schema={"type": "object", "class": "ProjectionResult"},
        tags=["geometry", "projection"],
        is_placeholder=True,
    )

    async def _arun(
        self,
        *,
        context: ToolContext,
        reconstruction: ReconstructionResult,
        box: BoundingBox2D | list[float] | tuple[float, ...],
        selected_index: int = 0,
        **kwargs: Any,
    ) -> ToolResult:
        coords = box.as_list() if isinstance(box, BoundingBox2D) else [float(item) for item in box]
        if len(coords) != 4:
            return self.error(f"box must contain 4 coordinates, got {coords}")
        image_size = None
        if 0 <= selected_index < len(reconstruction.camera_poses):
            image_size = reconstruction.camera_poses[selected_index].image_size
        if image_size is None:
            width = max(coords[0], coords[2], 1.0)
            height = max(coords[1], coords[3], 1.0)
        else:
            width, height = image_size
        cx, cy = bbox_center(coords)
        x = cx / max(width, 1.0) - 0.5
        y = cy / max(height, 1.0) - 0.5
        points = [
            [x, y, 1.0],
            [coords[0] / max(width, 1.0) - 0.5, coords[1] / max(height, 1.0) - 0.5, 1.0],
            [coords[2] / max(width, 1.0) - 0.5, coords[3] / max(height, 1.0) - 0.5, 1.0],
        ]
        result = ProjectionResult(
            points_3d=points,
            points_confidence=[0.1 for _ in points],
            selected_index=selected_index,
            metadata={
                "note": "Placeholder projection uses normalized image coordinates. Real backend should sample dense world_points.",
                "box": coords,
            },
        )
        return self.success(result)
