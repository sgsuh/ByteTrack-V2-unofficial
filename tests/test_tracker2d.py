import numpy as np

from bytetrack_v2.core.basetrack import BaseTrack
from bytetrack_v2.tracker2d.byte_tracker import BYTETracker2D


def box(x, y, score, w=40, h=100):
    return [x, y, x + w, y + h, score]


def run(tracker, frames):
    BaseTrack.reset_id()
    return [{t.track_id: t.tlbr for t in tracker.update(np.array(f))} for f in frames]


def test_ids_are_stable_for_moving_objects():
    frames = [[box(100 + 5 * t, 100, 0.9), box(400 - 5 * t, 120, 0.85)] for t in range(10)]
    outputs = run(BYTETracker2D(), frames)
    assert set(outputs[-1]) == {1, 2}
    for out in outputs[1:]:
        assert set(out) == {1, 2}


def test_low_score_detection_keeps_track_with_byte():
    frames = [[box(100 + 5 * t, 100, 0.9)] for t in range(5)]
    frames += [[box(125 + 5 * t, 100, 0.3)] for t in range(3)]  # occluded: score drops
    frames += [[box(140 + 5 * t, 100, 0.9)] for t in range(3)]
    with_byte = run(BYTETracker2D(use_byte=True), frames)
    without_byte = run(BYTETracker2D(use_byte=False), frames)
    # BYTE recovers the object in the low-score frames under the same id.
    assert all(list(out) == [1] for out in with_byte[1:])
    assert all(len(out) == 0 for out in without_byte[5:8])


def test_lost_track_is_reborn_with_same_id():
    frames = [[box(100, 100, 0.9)] for _ in range(5)] + [[] for _ in range(5)]
    frames += [[box(100, 100, 0.9)]]
    tracker = BYTETracker2D(track_buffer=30)
    BaseTrack.reset_id()
    outs = [{t.track_id for t in tracker.update(np.array(f).reshape(-1, 5))} for f in frames]
    assert outs[-1] == {1}


def test_first_frame_tracks_are_activated_immediately():
    tracker = BYTETracker2D()
    BaseTrack.reset_id()
    assert len(tracker.update(np.array([box(0, 0, 0.9)]))) == 1
    # Tracks born later are unconfirmed until matched again.
    assert len(tracker.update(np.array([box(0, 0, 0.9), box(300, 0, 0.9)]))) == 1
