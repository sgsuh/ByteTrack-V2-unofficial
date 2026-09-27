import numpy as np

from bytetrack_v2.tracker3d.giou3d import giou_3d

UNIT = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]


def shifted(dx=0.0, dy=0.0, dz=0.0, yaw=0.0):
    return [dx, dy, dz, yaw, 1.0, 1.0, 1.0]


def test_identical_boxes():
    giou, iou = giou_3d([UNIT], [UNIT], return_iou=True)
    np.testing.assert_allclose(giou, [[1.0]])
    np.testing.assert_allclose(iou, [[1.0]])


def test_half_overlap_along_x():
    # inter = 0.5, union = 1.5, enclosing = 1.5 -> GIoU = IoU = 1/3
    np.testing.assert_allclose(giou_3d([UNIT], [shifted(dx=0.5)]), [[1 / 3]])


def test_disjoint_boxes_are_negative():
    # inter = 0, union = 2, enclosing = 3 -> GIoU = -1/3
    np.testing.assert_allclose(giou_3d([UNIT], [shifted(dx=2.0)]), [[-1 / 3]])


def test_vertical_offset():
    # Half vertical overlap: inter = 0.5, union = 1.5, enclosing = 1.5
    np.testing.assert_allclose(giou_3d([UNIT], [shifted(dz=0.5)]), [[1 / 3]])


def test_yaw_symmetry_of_square():
    np.testing.assert_allclose(giou_3d([UNIT], [shifted(yaw=np.pi / 2)]), [[1.0]], atol=1e-9)


def test_rotated_rectangle_overlap():
    # A 2x1 box vs the same box rotated 90 deg: intersection is the 1x1 center.
    a = [0, 0, 0, 0.0, 2.0, 1.0, 1.0]
    b = [0, 0, 0, np.pi / 2, 2.0, 1.0, 1.0]
    giou, iou = giou_3d([a], [b], return_iou=True)
    np.testing.assert_allclose(iou, [[1 / 3]], atol=1e-9)
    # Enclosing hull is an octagon: 2x2 square minus four 0.5-leg corner triangles
    hull_area = 4 - 4 * (0.5 * 0.5 * 0.5)
    np.testing.assert_allclose(giou, [[1 / 3 - (hull_area - 3) / hull_area]], atol=1e-9)


def test_shapes_and_empty():
    out = giou_3d(np.array([UNIT] * 3), np.array([UNIT, shifted(dx=5)]))
    assert out.shape == (3, 2)
    assert giou_3d(np.zeros((0, 7)), [UNIT]).shape == (0, 1)
