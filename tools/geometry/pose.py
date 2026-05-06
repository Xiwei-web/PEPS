"""Object pose estimation tool slot."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from peps.tools.base import BaseTool, LocalModelRequirement, ToolContext, ToolResult, ToolSpec
from peps.tools.geometry.detection import BoundingBox2D
from peps.tools.geometry.geometry_ops import centroid, identity_matrix4, normalize
from peps.tools.geometry.projection import ProjectionResult
from peps.tools.geometry.reconstruction import ReconstructionResult


@dataclass(slots=True)
class ObjectPoseResult:
    T_obj2world: list[list[float]]
    centroid: list[float]
    axes: dict[str, list[float]]
    label: str = "object"
    obb: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_message_content(self) -> str:
        return f"ObjectPoseResult: pose estimated for {self.label!r} using {self.metadata.get('backend', 'placeholder')}."


class PredictObjectPoseTool(BaseTool):
    spec = ToolSpec(
        name="predict_obj_pose",
        description="Estimate object 6DoF pose and semantic orientation from reconstruction and a 2D box.",
        input_schema={
            "type": "object",
            "required": ["reconstruction", "box"],
            "properties": {
                "reconstruction": {"type": "object"},
                "box": {"type": "array"},
                "selected_index": {"type": "integer"},
                "obj_label": {"type": "string"},
                "projection": {"type": "object"},
            },
        },
        output_schema={"type": "object", "class": "ObjectPoseResult"},
        model_requirements=[
            LocalModelRequirement(
                name="OrientationAnything",
                required_for="semantic object orientation and object-based reference frames",
                local_path_env="PEPS_ORIENTATION_ANYTHING_MODEL_DIR",
                package="orientation-anything",
                download_hint="Download OrientationAnything weights locally and set PEPS_ORIENTATION_ANYTHING_MODEL_DIR.",
                required=False,
            ),
            LocalModelRequirement(
                name="Open3D",
                required_for="oriented bounding box cleanup and point-cloud operations",
                package="open3d",
                required=False,
            ),
        ],
        tags=["geometry", "pose", "orientation"],
        is_placeholder=True,
    )

    async def _arun(
        self,
        *,
        context: ToolContext,
        reconstruction: ReconstructionResult,
        box: BoundingBox2D | list[float] | tuple[float, ...],
        selected_index: int = 0,
        obj_label: str | None = None,
        projection: ProjectionResult | None = None,
        orientation_axis: list[float] | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        if projection is None:
            points = [[0.0, 0.0, 1.0]]
        else:
            points = projection.points_3d
        center = centroid(points)
        z_axis = normalize(orientation_axis or [0.0, 0.0, 1.0])
        y_axis = [0.0, 1.0, 0.0]
        x_axis = [1.0, 0.0, 0.0]
        matrix = identity_matrix4()
        matrix[0][3], matrix[1][3], matrix[2][3] = center[:3]
        result = ObjectPoseResult(
            T_obj2world=matrix,
            centroid=center,
            axes={"+X": x_axis, "+Y": y_axis, "+Z": z_axis},
            label=obj_label or (box.label if isinstance(box, BoundingBox2D) else "object"),
            metadata={
                "backend": "placeholder",
                "selected_index": selected_index,
                "note": "Real backend should combine projected 3D points with OrientationAnything output.",
            },
        )
        return self.success(result)

