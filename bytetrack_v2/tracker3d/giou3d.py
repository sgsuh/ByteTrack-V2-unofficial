"""3D (generalized) IoU for yaw-rotated boxes.

Boxes follow the ByteTrackV2 state ordering `[x, y, z, yaw, l, w, h]`, where
(x, y, z) is the box center in world coordinates, `yaw` is the heading around
the z-axis, `l` is the extent along the heading and `w` the lateral extent.

Boxes are assumed to rotate only around z (true for nuScenes), so the 3D
intersection is the BEV polygon intersection times the vertical overlap.
"""
import numpy as np
import shapely


def bev_corners(boxes):
    """(N, 7) boxes -> (N, 4, 2) BEV corners in counter-clockwise order."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 7)
    x, y, yaw, l, w = boxes[:, 0], boxes[:, 1], boxes[:, 3], boxes[:, 4], boxes[:, 5]
    local = np.array([[0.5, 0.5], [-0.5, 0.5], [-0.5, -0.5], [0.5, -0.5]])
    dx = local[None, :, 0] * l[:, None]
    dy = local[None, :, 1] * w[:, None]
    cos, sin = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
    cx = x[:, None] + dx * cos - dy * sin
    cy = y[:, None] + dx * sin + dy * cos
    return np.stack([cx, cy], axis=-1)


def giou_3d(boxes_a, boxes_b, return_iou=False):
    """Pairwise 3D GIoU between (N, 7) and (M, 7) boxes -> (N, M) in [-1, 1].

    GIoU = IoU - (|C| - |A u B|) / |C|, where C is the smallest enclosing volume
    (convex hull of both BEV footprints times the joint vertical extent).
    """
    a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 7)
    b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 7)
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        empty = np.zeros((n, m), dtype=np.float64)
        return (empty, empty.copy()) if return_iou else empty

    corners_a, corners_b = bev_corners(a), bev_corners(b)
    poly_a, poly_b = shapely.polygons(corners_a), shapely.polygons(corners_b)
    inter_area = shapely.area(shapely.intersection(poly_a[:, None], poly_b[None, :]))

    bottom_a, top_a = a[:, 2] - a[:, 6] / 2, a[:, 2] + a[:, 6] / 2
    bottom_b, top_b = b[:, 2] - b[:, 6] / 2, b[:, 2] + b[:, 6] / 2
    h_inter = np.clip(np.minimum(top_a[:, None], top_b[None, :])
                      - np.maximum(bottom_a[:, None], bottom_b[None, :]), 0, None)
    h_enclose = (np.maximum(top_a[:, None], top_b[None, :])
                 - np.minimum(bottom_a[:, None], bottom_b[None, :]))

    vol_a = a[:, 4] * a[:, 5] * a[:, 6]
    vol_b = b[:, 4] * b[:, 5] * b[:, 6]
    inter_vol = inter_area * h_inter
    union_vol = vol_a[:, None] + vol_b[None, :] - inter_vol

    pair_corners = np.concatenate([
        np.broadcast_to(corners_a[:, None], (n, m, 4, 2)),
        np.broadcast_to(corners_b[None, :], (n, m, 4, 2)),
    ], axis=2).reshape(n * m, 8, 2)
    hull_area = shapely.area(shapely.convex_hull(shapely.multipoints(pair_corners))).reshape(n, m)
    enclose_vol = hull_area * h_enclose

    eps = 1e-9
    iou = inter_vol / np.maximum(union_vol, eps)
    giou = iou - (enclose_vol - union_vol) / np.maximum(enclose_vol, eps)
    return (giou, iou) if return_iou else giou
