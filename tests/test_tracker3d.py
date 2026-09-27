import numpy as np
import pytest

from bytetrack_v2.core.basetrack import BaseTrack
from bytetrack_v2.tracker3d.byte_tracker import MOTION_MODES, BYTETracker3D
from bytetrack_v2.tracker3d.kalman_filter import KalmanFilter3D, align_yaw

DT = 0.5


def car(x, y, yaw=0.0):
    return [x, y, 0.0, yaw, 4.5, 1.9, 1.7]


def step(tracker, dets):
    """dets: list of (box, score, velocity)."""
    boxes = np.array([d[0] for d in dets]).reshape(-1, 7)
    scores = np.array([d[1] for d in dets])
    vels = np.array([d[2] for d in dets]).reshape(-1, 2)
    return {t.track_id for t in tracker.update(boxes, scores, vels, DT)}


@pytest.mark.parametrize("motion", MOTION_MODES)
def test_constant_velocity_ids_are_stable(motion):
    BaseTrack.reset_id()
    tracker = BYTETracker3D(motion=motion, giou_thresh=-0.1)
    v = 10.0  # 5 m per frame, larger than the car width
    for t in range(12):
        ids = step(tracker, [(car(v * DT * t, 0.0), 0.9, [v, 0.0]),
                             (car(-v * DT * t, 6.0, np.pi), 0.8, [-v, 0.0])])
        if t > 0:
            assert ids == {1, 2}


def test_velocity_backward_prediction_handles_abrupt_motion():
    """A sudden lateral jump is followed via the detected velocity, not the KF."""
    BaseTrack.reset_id()
    trackers = {m: BYTETracker3D(motion=m, giou_thresh=-0.1) for m in ("kalman", "integrated")}
    ids = {}
    for x in [0.0, 5.0, 10.0, 15.0]:
        for m, tr in trackers.items():
            ids[m] = step(tr, [(car(x, 0.0), 0.9, [10.0, 0.0])])
    # Abrupt turn: the object moves 5 m in y and the detector reports it.
    jump = [(car(17.0, 5.0), 0.9, [4.0, 10.0])]
    assert step(trackers["integrated"], jump) == ids["integrated"]
    assert not (step(trackers["kalman"], jump) & ids["kalman"])


def test_lost_track_rebirth_uses_kalman_forward_prediction():
    BaseTrack.reset_id()
    tracker = BYTETracker3D(motion="integrated", giou_thresh=-0.1)
    v = 10.0
    for t in range(6):
        step(tracker, [(car(v * DT * t, 0.0), 0.9, [v, 0.0])])
    for _ in range(4):  # disappears for 2 s
        step(tracker, [])
    # Reappears where the KF expects it; detected velocity is noisy/zero.
    ids = step(tracker, [(car(v * DT * 10, 0.0), 0.9, [0.0, 0.0])])
    assert ids == {1}


def test_low_score_detection_second_association():
    BaseTrack.reset_id()
    with_byte = BYTETracker3D(use_byte=True, track_thresh=0.5, low_thresh=0.1)
    without = BYTETracker3D(use_byte=False, track_thresh=0.5, low_thresh=0.1)
    for tr in (with_byte, without):
        BaseTrack.reset_id()
        for t in range(3):
            step(tr, [(car(t, 0.0), 0.9, [2.0, 0.0])])
    assert step(with_byte, [(car(3, 0.0), 0.3, [2.0, 0.0])]) == {1}
    assert step(without, [(car(3, 0.0), 0.3, [2.0, 0.0])]) == set()


def test_align_yaw_flips_reversed_heading():
    assert np.isclose(align_yaw(np.pi, 0.05), 0.0)
    assert np.isclose(align_yaw(0.3, 0.0), 0.3)
    assert np.isclose(align_yaw(-np.pi + 0.1, np.pi - 0.1), np.pi + 0.1)


def test_kf3d_estimates_velocity():
    kf = KalmanFilter3D()
    mean, cov = kf.initiate(np.array(car(0.0, 0.0)))
    for t in range(1, 15):
        mean, cov = kf.predict(mean, cov, DT)
        mean, cov = kf.update(mean, cov, np.array(car(3.0 * DT * t, 1.0 * DT * t)))
    np.testing.assert_allclose(mean[7:9], [3.0, 1.0], atol=0.2)
    assert np.all(np.linalg.eigvalsh(cov) > 0)
