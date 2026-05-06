"""Primitive IDs and lightweight schema-specific constants."""

from __future__ import annotations

from enum import StrEnum

DEFAULT_SCHEMA_VERSION = "0.1.0"


class FramePrimitiveID(StrEnum):
    CAMERA_BASED = "frame.camera_based"
    OBJECT_BASED = "frame.object_based"
    DIRECTION_BASED = "frame.direction_based"
    EVENT_DEFINED_DIRECTION = "frame.event_defined_direction"


class EntityPrimitiveID(StrEnum):
    TARGET = "entity.target"
    ANCHOR = "entity.anchor"
    CANDIDATE_SET = "entity.candidate_set"
    VIEW_SOURCE = "entity.view_source"
    FILTER = "entity.filter"
    COUNT = "entity.count"


class StatePrimitiveID(StrEnum):
    CAMERA_POSE = "state.camera_pose"
    CENTROID = "state.centroid"
    BBOX3D = "state.bbox3d"
    POINTSET = "state.pointset"
    POSE3D = "state.pose3d"
    ORIENTATION = "state.orientation"
    SURFACE_NORMAL = "state.surface_normal"


class MetricPrimitiveID(StrEnum):
    RELATIVE_DISPLACEMENT = "metric.relative_displacement"
    PLANAR_DISPLACEMENT = "metric.planar_displacement"
    VERTICAL_OFFSET = "metric.vertical_offset"
    EUCLIDEAN_DISTANCE = "metric.euclidean_distance"
    SURFACE_DISTANCE = "metric.surface_distance"
    SIZE_COMPARISON = "metric.size_comparison"
    ORIENTATION_DIFFERENCE = "metric.orientation_difference"
    FACING_ALIGNMENT = "metric.facing_alignment"
    CONTAINMENT = "metric.containment"
    BETWEEN_RATIO = "metric.between_ratio"
    COUNT = "metric.count"
    CAMERA_TRANSLATION = "metric.camera_translation"
    CAMERA_ROTATION = "metric.camera_rotation"
    REGION_OBJECT_QUERY = "metric.region_object_query"


FESM_ORDER = ("frame", "entity", "state", "metric")

CORE_TOOL_NAMES = (
    "reconstruct",
    "detect",
    "project_box_to_3d_points",
    "predict_obj_pose",
    "estimate_scale",
    "ocr",
    "analyze_motion",
    "code",
)

