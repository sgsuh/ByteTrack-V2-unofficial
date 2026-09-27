"""3D ByteTrackV2: BYTE association with complementary 3D motion prediction.

Motion prediction modes (ByteTrackV2 Sec. 3.3, Table 7):
    "kalman":     all tracks are forward-predicted by the Kalman filter to
                  frame t and compared with the detections D_t.
    "velocity":   detections are backward-predicted with their detected
                  velocity, D_hat_{t-1} = D_t - V_t * dt (Eq. 7), and compared
                  with the last track states T_{t-1}.
    "integrated": tracks alive at t-1 use the backward prediction (short-term
                  association); lost tracks use the Kalman forward prediction
                  (long-term association / track rebirth).

Similarity is 3D GIoU (Eq. 9); a pair is rejected when GIoU < `giou_thresh`.
"""
import numpy as np

from ..core.basetrack import BaseTrack, TrackState
from ..core.byte_tracker import BaseByteTracker
from .giou3d import giou_3d
from .kalman_filter import KalmanFilter3D

MOTION_MODES = ("kalman", "velocity", "integrated")


class STrack3D(BaseTrack):
    """A 3D detection / tracklet.

    Args:
        box: (7,) [x, y, z, yaw, l, w, h] in world coordinates.
        score: detection confidence.
        velocity: (2,) detected [vx, vy] in m/s (world frame).
        dt: time since the previous frame, used for the backward prediction.
    """

    def __init__(self, box, score, velocity, dt, kalman_filter, score_alpha=None, attrs=None):
        super().__init__(score)
        self.det_box = np.asarray(box, dtype=np.float64)
        self.det_velocity = np.asarray(velocity, dtype=np.float64)
        self.kalman_filter = kalman_filter
        self.score_alpha = score_alpha
        self.attrs = attrs or {}

        self.mean, self.covariance = None, None
        self.last_mean = None  # posterior at the previous frame (T_{t-1})

        self.back_box = self.det_box.copy()
        self.back_box[:2] -= self.det_velocity[:2] * dt

    # --------------------------------------------------------------- boxes
    @property
    def box(self):
        """Current box: posterior/prediction for tracks, raw box for detections."""
        if self.mean is None:
            return self.det_box.copy()
        return self.mean[:7].copy()

    @property
    def last_box(self):
        """Track box at the previous frame, before this frame's prediction."""
        if self.last_mean is None:
            return self.box
        return self.last_mean[:7].copy()

    @property
    def velocity(self):
        return self.mean[7:10].copy() if self.mean is not None else np.r_[self.det_velocity[:2], 0.0]

    # ---------------------------------------------------------- life cycle
    def predict(self, dt):
        self.last_mean = self.mean.copy()
        self.mean, self.covariance = self.kalman_filter.predict(self.mean, self.covariance, dt)

    def activate(self, frame_id):
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.det_box, self.det_velocity)
        self.last_mean = self.mean.copy()
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = frame_id == 1
        self.frame_id = frame_id
        self.start_frame = frame_id

    def _kf_update(self, det):
        use_score = self.score_alpha is not None
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, det.det_box,
            score=det.score if use_score else None, alpha=self.score_alpha)
        self.score = det.score
        self.attrs = det.attrs

    def re_activate(self, det, frame_id):
        self._kf_update(det)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id

    def update(self, det, frame_id):
        self._kf_update(det)
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.state = TrackState.Tracked
        self.is_activated = True

    def __repr__(self):
        return f"STrack3D_{self.track_id}({self.start_frame}-{self.end_frame})"


class BYTETracker3D(BaseByteTracker):
    """Single-class 3D ByteTrackV2 tracker.

    Unconfirmed tracks are motion-predicted like all other tracks, since
    frames are far apart (0.5 s on nuScenes) and skipping the prediction
    would corrupt the velocity estimate.

    Args:
        track_thresh: high/low split threshold tau (0.2 for CenterPoint / PETRv2).
        giou_thresh: minimum 3D GIoU of a valid match (class specific, Sec. 5.1).
        max_time_lost: frames a lost track is kept (30).
        motion: one of MOTION_MODES.
        score_alpha: alpha of the score-adaptive measurement noise (Eq. 10); None disables.
        use_byte: associate low-score detections in a second stage.
        low_thresh: detections below this score are discarded.
        new_track_thresh: minimum score to start a track (defaults to tau).
        kf_params: keyword arguments of KalmanFilter3D.
    """

    predict_unconfirmed = True

    def __init__(self, track_thresh=0.2, giou_thresh=-0.5, max_time_lost=30, motion="integrated",
                 score_alpha=10.0, use_byte=True, low_thresh=0.0, new_track_thresh=None,
                 kf_params=None):
        if motion not in MOTION_MODES:
            raise ValueError(f"motion must be one of {MOTION_MODES}, got {motion!r}")
        match_thresh = 1.0 - giou_thresh  # cost = 1 - GIoU in [0, 2]
        super().__init__(
            track_thresh=track_thresh,
            low_thresh=low_thresh,
            new_track_thresh=track_thresh if new_track_thresh is None else new_track_thresh,
            max_time_lost=max_time_lost,
            match_thresh=match_thresh,
            second_match_thresh=match_thresh,
            unconfirmed_match_thresh=match_thresh,
            use_byte=use_byte,
        )
        self.motion = motion
        self.score_alpha = score_alpha
        self.kalman_filter = KalmanFilter3D(**(kf_params or {}))
        self.dt = 0.5

    def update(self, boxes, scores, velocities, dt, attrs=None):
        """Track one frame of detections of this class.

        Args:
            boxes: (N, 7) [x, y, z, yaw, l, w, h] world-frame boxes.
            scores: (N,) detection scores.
            velocities: (N, 2) detected velocities [vx, vy] in m/s.
            dt: seconds elapsed since the previous frame.
            attrs: optional list of per-detection dicts carried to the output.
        """
        self.dt = dt
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 7)
        attrs = attrs if attrs is not None else [None] * len(boxes)
        dets = [STrack3D(b, s, v, dt, self.kalman_filter, self.score_alpha, a)
                for b, s, v, a in zip(boxes, scores, velocities, attrs)]
        return super().update(dets)

    # ---------------------------------------------------------------- hooks
    def predict(self, tracks):
        for t in tracks:
            t.predict(self.dt)

    def first_cost(self, tracks, detections):
        if not tracks or not detections:
            return np.zeros((len(tracks), len(detections)))
        if self.motion == "kalman":
            return 1.0 - giou_3d([t.box for t in tracks], [d.det_box for d in detections])
        if self.motion == "velocity":
            return 1.0 - giou_3d([t.last_box for t in tracks], [d.back_box for d in detections])

        # Integrated: backward prediction for alive tracks, forward (KF) for lost ones.
        cost = np.empty((len(tracks), len(detections)))
        alive = np.array([t.state == TrackState.Tracked for t in tracks])
        if alive.any():
            alive_tracks = [t for t, a in zip(tracks, alive) if a]
            cost[alive] = 1.0 - giou_3d([t.last_box for t in alive_tracks],
                                        [d.back_box for d in detections])
        if (~alive).any():
            lost_tracks = [t for t, a in zip(tracks, alive) if not a]
            cost[~alive] = 1.0 - giou_3d([t.box for t in lost_tracks],
                                         [d.det_box for d in detections])
        return cost


class MultiClassTracker3D:
    """Runs one BYTETracker3D per class; association never crosses classes.

    Args:
        classes: class names to track.
        giou_thresh: a float, or a dict of per-class GIoU thresholds.
        **kwargs: forwarded to every BYTETracker3D.
    """

    def __init__(self, classes, giou_thresh=-0.5, **kwargs):
        self.trackers = {}
        for c in classes:
            thr = giou_thresh[c] if isinstance(giou_thresh, dict) else giou_thresh
            self.trackers[c] = BYTETracker3D(giou_thresh=thr, **kwargs)

    def update(self, frame_dets, dt):
        """Track one frame.

        Args:
            frame_dets: {class: dict(boxes=(N, 7), scores=(N,), velocities=(N, 2), attrs=list)}.
                Classes without detections may be omitted.
            dt: seconds since the previous frame.

        Returns:
            {class: list of active STrack3D}
        """
        out = {}
        for c, tracker in self.trackers.items():
            d = frame_dets.get(c)
            if d is None:
                d = dict(boxes=np.zeros((0, 7)), scores=np.zeros(0), velocities=np.zeros((0, 2)), attrs=[])
            out[c] = tracker.update(d["boxes"], d["scores"], d["velocities"], dt, d.get("attrs"))
        return out
