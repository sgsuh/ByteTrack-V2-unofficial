"""Track life-cycle primitives shared by the 2D and 3D trackers.

Adapted from ByteTrack (https://github.com/FoundationVision/ByteTrack, MIT).
"""
import itertools


class TrackState:
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack:
    _id_counter = itertools.count(1)

    def __init__(self, score):
        self.track_id = 0
        self.is_activated = False
        self.state = TrackState.New
        self.score = score
        self.start_frame = 0
        self.frame_id = 0
        self.tracklet_len = 0

    @property
    def end_frame(self):
        return self.frame_id

    @staticmethod
    def next_id():
        return next(BaseTrack._id_counter)

    @staticmethod
    def reset_id():
        BaseTrack._id_counter = itertools.count(1)

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed


def joint_tracks(tlista, tlistb):
    """Union of two track lists, keyed by track id (order-preserving)."""
    exists = set()
    res = []
    for t in itertools.chain(tlista, tlistb):
        if t.track_id not in exists:
            exists.add(t.track_id)
            res.append(t)
    return res


def sub_tracks(tlista, tlistb):
    """Tracks of `tlista` whose ids are not in `tlistb`."""
    ids_b = {t.track_id for t in tlistb}
    return [t for t in tlista if t.track_id not in ids_b]
