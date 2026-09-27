import numpy as np

from bytetrack_v2.core.matching import box_iou_2d, fuse_score, linear_assignment


def test_linear_assignment_respects_threshold():
    cost = np.array([[0.1, 0.9], [0.8, 0.95]])
    matches, ua, ub = linear_assignment(cost, thresh=0.5)
    assert matches.tolist() == [[0, 0]]
    assert ua.tolist() == [1]
    assert ub.tolist() == [1]


def test_linear_assignment_empty():
    matches, ua, ub = linear_assignment(np.zeros((0, 3)), thresh=0.5)
    assert matches.shape == (0, 2)
    assert len(ua) == 0 and ub.tolist() == [0, 1, 2]


def test_box_iou_2d_pixel_inclusive():
    # Two 10x10 (pixel-inclusive) boxes overlapping in a 5x10 region.
    a = [[0, 0, 9, 9]]
    b = [[5, 0, 14, 9], [100, 100, 110, 110]]
    iou = box_iou_2d(a, b)
    np.testing.assert_allclose(iou, [[50 / 150, 0.0]])


def test_fuse_score():
    class Det:
        def __init__(self, score):
            self.score = score

    cost = np.array([[0.2, 1.0]])
    fused = fuse_score(cost, [Det(0.5), Det(0.9)])
    np.testing.assert_allclose(fused, [[1 - 0.8 * 0.5, 1.0]])
