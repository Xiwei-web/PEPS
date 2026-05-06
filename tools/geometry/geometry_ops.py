"""Deterministic geometry helpers used by tools and generated code."""

from __future__ import annotations

from math import acos, sqrt
from typing import Iterable


Vector = list[float]
Matrix4 = list[list[float]]


def as_vector(value: Iterable[float]) -> Vector:
    return [float(item) for item in value]


def add(a: Iterable[float], b: Iterable[float]) -> Vector:
    return [x + y for x, y in zip(as_vector(a), as_vector(b), strict=True)]


def subtract(a: Iterable[float], b: Iterable[float]) -> Vector:
    return [x - y for x, y in zip(as_vector(a), as_vector(b), strict=True)]


def dot(a: Iterable[float], b: Iterable[float]) -> float:
    return sum(x * y for x, y in zip(as_vector(a), as_vector(b), strict=True))


def norm(a: Iterable[float]) -> float:
    return sqrt(dot(a, a))


def normalize(a: Iterable[float], *, eps: float = 1e-8) -> Vector:
    vector = as_vector(a)
    length = norm(vector)
    if length < eps:
        return [0.0 for _ in vector]
    return [item / length for item in vector]


def centroid(points: Iterable[Iterable[float]]) -> Vector:
    rows = [as_vector(point) for point in points]
    if not rows:
        return [0.0, 0.0, 0.0]
    dims = len(rows[0])
    return [sum(row[i] for row in rows) / len(rows) for i in range(dims)]


def euclidean_distance(a: Iterable[float], b: Iterable[float]) -> float:
    return norm(subtract(a, b))


def angle_between(a: Iterable[float], b: Iterable[float], *, degrees: bool = True) -> float:
    na = normalize(a)
    nb = normalize(b)
    value = max(-1.0, min(1.0, dot(na, nb)))
    radians = acos(value)
    return radians * 180.0 / 3.141592653589793 if degrees else radians


def identity_matrix4() -> Matrix4:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def transform_point(matrix: Matrix4, point: Iterable[float]) -> Vector:
    x, y, z = as_vector(point)[:3]
    vec = [x, y, z, 1.0]
    out = [sum(matrix[row][col] * vec[col] for col in range(4)) for row in range(4)]
    scale = out[3] if abs(out[3]) > 1e-8 else 1.0
    return [out[0] / scale, out[1] / scale, out[2] / scale]


def bbox_center(box: Iterable[float]) -> list[float]:
    x1, y1, x2, y2 = as_vector(box)
    return [(x1 + x2) / 2.0, (y1 + y2) / 2.0]


def bbox_size(box: Iterable[float]) -> list[float]:
    x1, y1, x2, y2 = as_vector(box)
    return [abs(x2 - x1), abs(y2 - y1)]


def planar_displacement(
    entity_a: Iterable[float],
    entity_b: Iterable[float],
    *,
    horizontal_axes: tuple[int, int] = (0, 2),
) -> Vector:
    delta = subtract(entity_a, entity_b)
    return [delta[horizontal_axes[0]], delta[horizontal_axes[1]]]


def vertical_offset(entity_a: Iterable[float], entity_b: Iterable[float], *, vertical_axis: int = 1) -> float:
    return subtract(entity_a, entity_b)[vertical_axis]

