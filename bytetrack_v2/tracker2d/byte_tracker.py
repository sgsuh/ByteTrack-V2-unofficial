"""2D ByteTrack: BYTE association with an image-plane Kalman filter and IoU.

Behavior follows ByteTrack's `BYTETracker`
(https://github.com/FoundationVision/ByteTrack, MIT) so that the published
MOT17 numbers can be reproduced.
"""
import numpy as np

from ..core import matching
from ..core.basetrack import BaseTrack, TrackState
from ..core.byte_tracker import BaseByteTracker
from .kalman_filter import KalmanFilter2D


class STrack2D(BaseTrack):
    def __init__(self, tlwh, score, kalman_filter, score_alpha=None):
        super().__init__(score)
        self._tlwh = np.asarray(tlwh, dtype=np.float64)
        self.kalman_filter = kalman_filter
        self.score_alpha = score_alpha
        self.mean, self.covariance = None, None

    @staticmethod
    def multi_predict(tracks, kalman_filter):
        if not tracks:
            return
        means = np.stack([t.mean.copy() for t in tracks])
        covs = np.stack([t.covariance for t in tracks])
        for i, t in enumerate(tracks):
            if t.state != TrackState.Tracked:
                means[i, 7] = 0  # freeze height velocity of lost tracks
        means, covs = kalman_filter.multi_predict(means, covs)
        for t, m, c in zip(tracks, means, covs):
            t.mean, t.covariance = m, c

    def activate(self, frame_id):
        """Start a new tracklet."""
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = frame_id == 1
        self.frame_id = frame_id
        self.start_frame = frame_id

    def _kf_update(self, det):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(det.tlwh),
            score=det.score if self.score_alpha is not None else None, alpha=self.score_alpha)

    def re_activate(self, det, frame_id):
        self._kf_update(det)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.score = det.score

    def update(self, det, frame_id):
        self.frame_id = frame_id
        self.tracklet_len += 1
        self._kf_update(det)
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = det.score

    @property
    def tlwh(self):
        """Box as (top-left x, top-left y, width, height)."""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Box as (min x, min y, max x, max y)."""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        ret = np.asarray(tlwh, dtype=np.float64).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr, dtype=np.float64).copy()
        ret[2:] -= ret[:2]
        return ret

    def __repr__(self):
        return f"STrack2D_{self.track_id}({self.start_frame}-{self.end_frame})"


class BYTETracker2D(BaseByteTracker):
    """ByteTrack for 2D MOT.

    Args:
        track_thresh: high/low split threshold tau (paper default 0.6).
        match_thresh: first-association cost threshold (1 - IoU * score).
        track_buffer: frames to keep lost tracks at 30 FPS.
        frame_rate: sequence frame rate (scales `track_buffer`).
        fuse_score: fuse detection score into IoU cost (disabled for MOT20).
        score_alpha: if set, scale KF measurement noise by alpha * (1 - s)^2.
        use_byte: associate low-score detections (the BYTE second stage).
    """

    def __init__(self, track_thresh=0.6, match_thresh=0.9, track_buffer=30, frame_rate=30,
                 fuse_score=True, score_alpha=None, use_byte=True, low_thresh=0.1,
                 second_match_thresh=0.5, unconfirmed_match_thresh=0.7, duplicate_thresh=0.15):
        super().__init__(
            track_thresh=track_thresh,
            low_thresh=low_thresh,
            new_track_thresh=track_thresh + 0.1,
            max_time_lost=int(frame_rate / 30.0 * track_buffer),
            match_thresh=match_thresh,
            second_match_thresh=second_match_thresh,
            unconfirmed_match_thresh=unconfirmed_match_thresh,
            use_byte=use_byte,
        )
        self.fuse_score = fuse_score
        self.score_alpha = score_alpha
        self.duplicate_thresh = duplicate_thresh
        self.kalman_filter = KalmanFilter2D()

    def update(self, detections):
        """Track one frame.

        Args:
            detections: (N, 5) array of [x1, y1, x2, y2, score] in image pixels.
        """
        detections = np.asarray(detections, dtype=np.float64).reshape(-1, 5)
        dets = [STrack2D(STrack2D.tlbr_to_tlwh(d[:4]), d[4], self.kalman_filter, self.score_alpha)
                for d in detections]
        return super().update(dets)

    def predict(self, tracks):
        STrack2D.multi_predict(tracks, self.kalman_filter)

    def first_cost(self, tracks, detections):
        cost = matching.iou_distance(tracks, detections)
        return matching.fuse_score(cost, detections) if self.fuse_score else cost

    def second_cost(self, tracks, detections):
        return matching.iou_distance(tracks, detections)

    def remove_duplicates(self, tracked, lost):
        pdist = matching.iou_distance(tracked, lost)
        dup_a, dup_b = set(), set()
        for p, q in zip(*np.where(pdist < self.duplicate_thresh)):
            age_p = tracked[p].frame_id - tracked[p].start_frame
            age_q = lost[q].frame_id - lost[q].start_frame
            if age_p > age_q:
                dup_b.add(q)
            else:
                dup_a.add(p)
        return ([t for i, t in enumerate(tracked) if i not in dup_a],
                [t for i, t in enumerate(lost) if i not in dup_b])
