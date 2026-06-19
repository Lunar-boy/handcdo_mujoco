from __future__ import annotations

Point2D = tuple[float, float]


def polygon_area(vertices: tuple[Point2D, ...] | list[Point2D]) -> float:
    """Return the signed area of a 2D polygon."""
    if len(vertices) < 3:
        return 0.0
    return 0.5 * sum(
        x0 * y1 - x1 * y0
        for (x0, y0), (x1, y1) in zip(
            vertices,
            vertices[1:] + vertices[:1],
            strict=True,
        )
    )


def is_point_in_convex_polygon(
    point: Point2D,
    polygon: tuple[Point2D, ...] | list[Point2D],
    *,
    eps: float = 1e-9,
) -> bool:
    """Return whether a point lies inside or on a convex polygon."""
    clip = _counterclockwise(polygon)
    return all(
        _cross(start, end, point) >= -eps
        for start, end in zip(clip, clip[1:] + clip[:1], strict=True)
    )


def clip_convex_polygon(
    subject: tuple[Point2D, ...] | list[Point2D],
    clip_polygon: tuple[Point2D, ...] | list[Point2D],
    *,
    eps: float = 1e-9,
) -> tuple[Point2D, ...]:
    """Clip a polygon against a convex polygon with Sutherland-Hodgman."""
    output = list(_counterclockwise(subject))
    clip = _counterclockwise(clip_polygon)
    if len(output) < 3 or len(clip) < 3:
        return ()

    for clip_start, clip_end in zip(clip, clip[1:] + clip[:1], strict=True):
        input_vertices = output
        output = []
        if not input_vertices:
            break
        previous = input_vertices[-1]
        previous_inside = _cross(clip_start, clip_end, previous) >= -eps
        for current in input_vertices:
            current_inside = _cross(clip_start, clip_end, current) >= -eps
            if current_inside:
                if not previous_inside:
                    output.append(
                        _line_intersection(previous, current, clip_start, clip_end, eps)
                    )
                output.append(current)
            elif previous_inside:
                output.append(
                    _line_intersection(previous, current, clip_start, clip_end, eps)
                )
            previous = current
            previous_inside = current_inside
        output = _deduplicate(output, eps)

    if len(output) < 3 or abs(polygon_area(output)) <= eps:
        return ()
    return tuple(output)


def _counterclockwise(
    vertices: tuple[Point2D, ...] | list[Point2D],
) -> list[Point2D]:
    result = list(vertices)
    if polygon_area(result) < 0.0:
        result.reverse()
    return result


def _cross(start: Point2D, end: Point2D, point: Point2D) -> float:
    return (
        (end[0] - start[0]) * (point[1] - start[1])
        - (end[1] - start[1]) * (point[0] - start[0])
    )


def _line_intersection(
    segment_start: Point2D,
    segment_end: Point2D,
    line_start: Point2D,
    line_end: Point2D,
    eps: float,
) -> Point2D:
    segment_x = segment_end[0] - segment_start[0]
    segment_y = segment_end[1] - segment_start[1]
    line_x = line_end[0] - line_start[0]
    line_y = line_end[1] - line_start[1]
    denominator = segment_x * line_y - segment_y * line_x
    if abs(denominator) <= eps:
        return segment_end
    offset_x = line_start[0] - segment_start[0]
    offset_y = line_start[1] - segment_start[1]
    t = (offset_x * line_y - offset_y * line_x) / denominator
    return (
        segment_start[0] + t * segment_x,
        segment_start[1] + t * segment_y,
    )


def _deduplicate(vertices: list[Point2D], eps: float) -> list[Point2D]:
    result: list[Point2D] = []
    for vertex in vertices:
        if not result or _distance_sq(result[-1], vertex) > eps**2:
            result.append(vertex)
    if len(result) > 1 and _distance_sq(result[0], result[-1]) <= eps**2:
        result.pop()
    return result


def _distance_sq(first: Point2D, second: Point2D) -> float:
    return (first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2
