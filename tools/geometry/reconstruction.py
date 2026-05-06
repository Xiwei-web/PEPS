"""3D reconstruction tool slot.

GCA implements this with VGGT plus optional multi-view alignment. PEPS keeps the
same API shape but starts with a lightweight placeholder output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from peps.tools.base import BaseTool, LocalModelRequirement, ToolContext, ToolResult, ToolSpec
from peps.tools.geometry.geometry_ops import identity_matrix4


@dataclass(slots=True)
class CameraPose:
    view_id: str
    world_to_camera: list[list[float]]
    intrinsic: list[list[float]]
    image_size: tuple[int, int] | None = None


@dataclass(slots=True)
class ReconstructionResult:
    """Unified scene reconstruction in a camera-0 world frame."""

    camera_poses: list[CameraPose]
    world_points: Any | None = None
    world_points_confidence: Any | None = None
    reconstruction_type: str = "placeholder"
    reconstruction_model: str = "none"
    image_sources: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_message_content(self) -> str:
        return (
            f"ReconstructionResult(type={self.reconstruction_type}, "
            f"model={self.reconstruction_model}, views={len(self.camera_poses)})"
        )


class ReconstructionTool(BaseTool):
    spec = ToolSpec(
        name="reconstruct",
        description="Perform 3D reconstruction and estimate camera poses for one or more images.",
        input_schema={
            "type": "object",
            "required": ["image_sources"],
            "properties": {
                "image_sources": {"type": "array"},
                "user_question": {"type": "string"},
            },
        },
        output_schema={"type": "object", "class": "ReconstructionResult"},
        model_requirements=[
            LocalModelRequirement(
                name="VGGT",
                required_for="dense 3D reconstruction and camera pose estimation",
                local_path_env="PEPS_VGGT_MODEL_DIR",
                package="vggt",
                download_hint="Download VGGT weights locally and point PEPS_VGGT_MODEL_DIR to the checkpoint directory.",
                required=True,
            ),
            LocalModelRequirement(
                name="SceneAligner",
                required_for="single-image pair alignment when multi-view reconstruction is unreliable",
                package="open3d",
                download_hint="No separate checkpoint in the PEPS placeholder; real alignment may require Open3D and feature matching dependencies.",
                required=False,
            ),
        ],
        tags=["geometry", "reconstruction", "camera_pose"],
        is_placeholder=True,
    )

    async def _arun(
        self,
        *,
        context: ToolContext,
        image_sources: list[Any] | Any,
        user_question: str | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        if not isinstance(image_sources, list):
            image_sources = [image_sources]
        camera_poses: list[CameraPose] = []
        for index, source in enumerate(image_sources):
            size = _try_image_size(source)
            camera_poses.append(
                CameraPose(
                    view_id=f"view_{index}",
                    world_to_camera=identity_matrix4(),
                    intrinsic=_default_intrinsic(size),
                    image_size=size,
                )
            )
        result = ReconstructionResult(
            camera_poses=camera_poses,
            reconstruction_type="placeholder_single" if len(image_sources) == 1 else "placeholder_multiple",
            reconstruction_model="placeholder",
            image_sources=[str(item) for item in image_sources],
            metadata={
                "user_question": user_question,
                "note": "Install/configure VGGT backend to produce dense world_points.",
            },
        )
        return self.success(result)


def _try_image_size(source: Any) -> tuple[int, int] | None:
    if hasattr(source, "size") and isinstance(source.size, tuple):
        return source.size
    if isinstance(source, str):
        path = Path(source)
        if path.exists():
            try:
                from PIL import Image

                with Image.open(path) as image:
                    return image.size
            except Exception:
                return None
    return None


def _default_intrinsic(size: tuple[int, int] | None) -> list[list[float]]:
    width, height = size or (1, 1)
    focal = float(max(width, height))
    return [
        [focal, 0.0, width / 2.0],
        [0.0, focal, height / 2.0],
        [0.0, 0.0, 1.0],
    ]

