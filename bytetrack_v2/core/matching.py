"""Assignment and cost utilities.

Adapted from ByteTrack (https://github.com/FoundationVision/ByteTrack, MIT).
"""
import lap
import numpy as np


def linear_assignment(cost_matrix, thresh):
    """Solve the assignment problem, rejecting pairs whose cost exceeds `thresh`.

    Returns:
        matches: (K, 2) int array of (row, col) pairs.
        unmatched_a: 1-D int array of unmatched row indices.
        unmatched_b: 1-D int array of unmatched column indices.
    """
    cost_matrix = np.asarray(cost_matrix, dtype=np.float64)
    if cost_matrix.size == 0:
        return (np.empty((0, 2), dtype=int),
                np.arange(cost_matrix.shape[0]),
                np.arange(cost_matrix.shape[1]))
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    matches = np.asarray([[ix, mx] for ix, mx in enumerate(x) if mx >= 0], dtype=int).reshape(-1, 2)
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    return matches, unmatched_a, unmatched_b


def box_iou_2d(atlbrs, btlbrs):
    """Pairwise IoU of axis-aligned boxes in (x1, y1, x2, y2) format.

    Follows the `cython_bbox.bbox_overlaps` convention used by ByteTrack,
    where widths and heights include +1 (pixel-inclusive coordinates).
    """
    a = np.asarray(atlbrs, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(btlbrs, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)

    area_a = (a[:, 2] - a[:, 0] + 1) * (a[:, 3] - a[:, 1] + 1)
    area_b = (b[:, 2] - b[:, 0] + 1) * (b[:, 3] - b[:, 1] + 1)
    iw = np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]) + 1
    ih = np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]) + 1
    inter = np.clip(iw, 0, None) * np.clip(ih, 0, None)
    union = area_b[None, :] + area_a[:, None] - inter
    return np.where(inter > 0, inter / union, 0.0)


def iou_distance(atracks, btracks):
    """1 - IoU cost between two lists of objects exposing `.tlbr`."""
    atlbrs = [t.tlbr for t in atracks]
    btlbrs = [t.tlbr for t in btracks]
    return 1.0 - box_iou_2d(atlbrs, btlbrs)


def fuse_score(cost_matrix, detections):
    """Fuse detection confidence into an IoU cost: 1 - IoU * score."""
    if cost_matrix.size == 0:
        return cost_matrix
    det_scores = np.array([det.score for det in detections])[None, :]
    return 1.0 - (1.0 - cost_matrix) * det_scores
