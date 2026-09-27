"""Dimension-agnostic BYTE association (ByteTrackV2, Algorithm 1).

The control flow mirrors `BYTETracker.update` of ByteTrack
(https://github.com/FoundationVision/ByteTrack, MIT). Subclasses provide the
motion model and similarity through a small set of hooks, so that the same
hierarchical association drives both the 2D and the 3D trackers.
"""
from .basetrack import TrackState, joint_tracks, sub_tracks
from .matching import linear_assignment


class BaseByteTracker:
    """Hierarchical association of high- and low-score detections.

    Subclasses may set `predict_unconfirmed`; ByteTrack (2D) matches
    unconfirmed tracks without motion prediction.

    Args:
        track_thresh: tau; detections with score > tau are "high".
        low_thresh: detections with low_thresh < score < tau are "low".
        new_track_thresh: minimum score for initializing a new track.
        max_time_lost: frames a lost track is kept for re-association (rebirth).
        match_thresh: cost threshold of the first association.
        second_match_thresh: cost threshold of the second association.
        unconfirmed_match_thresh: cost threshold for unconfirmed tracks.
        use_byte: if False, low-score detections are discarded (ablation).
    """

    predict_unconfirmed = False

    def __init__(self, track_thresh, low_thresh, new_track_thresh, max_time_lost,
                 match_thresh, second_match_thresh, unconfirmed_match_thresh, use_byte=True):
        self.track_thresh = track_thresh
        self.low_thresh = low_thresh
        self.new_track_thresh = new_track_thresh
        self.max_time_lost = max_time_lost
        self.match_thresh = match_thresh
        self.second_match_thresh = second_match_thresh
        self.unconfirmed_match_thresh = unconfirmed_match_thresh
        self.use_byte = use_byte

        self.tracked_tracks = []
        self.lost_tracks = []
        self.removed_tracks = []
        self.frame_id = 0

    # ------------------------------------------------------------------ hooks
    def predict(self, tracks):
        """Motion prediction of `tracks` to the current frame (in place)."""
        raise NotImplementedError

    def first_cost(self, tracks, detections):
        """Cost matrix (len(tracks), len(detections)) of the first association."""
        raise NotImplementedError

    def second_cost(self, tracks, detections):
        """Cost matrix of the second association (low-score detections)."""
        return self.first_cost(tracks, detections)

    def unconfirmed_cost(self, tracks, detections):
        """Cost matrix between unconfirmed tracks and leftover high detections."""
        return self.first_cost(tracks, detections)

    def remove_duplicates(self, tracked, lost):
        """Resolve tracked/lost tracks that describe the same object."""
        return tracked, lost

    # ----------------------------------------------------------------- update
    def split_detections(self, detections):
        high = [d for d in detections if d.score > self.track_thresh]
        low = [d for d in detections if self.low_thresh < d.score < self.track_thresh]
        if not self.use_byte:
            low = []
        return high, low

    def update(self, detections):
        """Associate one frame of detections (track objects with `.score`).

        Returns the activated tracks in the `Tracked` state.
        """
        self.frame_id += 1
        activated, refound, lost, removed = [], [], [], []

        dets_high, dets_low = self.split_detections(detections)

        unconfirmed, tracked = [], []
        for track in self.tracked_tracks:
            (tracked if track.is_activated else unconfirmed).append(track)

        # First association: high-score detections vs. tracked + lost tracks.
        track_pool = joint_tracks(tracked, self.lost_tracks)
        self.predict(track_pool)
        if self.predict_unconfirmed:
            self.predict(unconfirmed)
        cost = self.first_cost(track_pool, dets_high)
        matches, u_track, u_det = linear_assignment(cost, thresh=self.match_thresh)
        self._apply_matches(matches, track_pool, dets_high, activated, refound)

        # Second association: low-score detections vs. remaining tracked tracks.
        r_tracked = [track_pool[i] for i in u_track if track_pool[i].state == TrackState.Tracked]
        cost = self.second_cost(r_tracked, dets_low)
        matches, u_track, _ = linear_assignment(cost, thresh=self.second_match_thresh)
        self._apply_matches(matches, r_tracked, dets_low, activated, refound)

        for i in u_track:
            track = r_tracked[i]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost.append(track)

        # Unconfirmed tracks (usually tracks with a single frame).
        dets_high = [dets_high[i] for i in u_det]
        cost = self.unconfirmed_cost(unconfirmed, dets_high)
        matches, u_unconfirmed, u_det = linear_assignment(cost, thresh=self.unconfirmed_match_thresh)
        for itrack, idet in matches:
            unconfirmed[itrack].update(dets_high[idet], self.frame_id)
            activated.append(unconfirmed[itrack])
        for i in u_unconfirmed:
            track = unconfirmed[i]
            track.mark_removed()
            removed.append(track)

        # Initialize new tracks from the remaining high-score detections.
        for i in u_det:
            det = dets_high[i]
            if det.score < self.new_track_thresh:
                continue
            det.activate(self.frame_id)
            activated.append(det)

        # Delete tracks that have been lost for too long.
        for track in self.lost_tracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed.append(track)

        self.tracked_tracks = [t for t in self.tracked_tracks if t.state == TrackState.Tracked]
        self.tracked_tracks = joint_tracks(self.tracked_tracks, activated)
        self.tracked_tracks = joint_tracks(self.tracked_tracks, refound)
        self.lost_tracks = sub_tracks(self.lost_tracks, self.tracked_tracks)
        self.lost_tracks.extend(lost)
        self.lost_tracks = sub_tracks(self.lost_tracks, self.removed_tracks)
        self.removed_tracks.extend(removed)
        self.tracked_tracks, self.lost_tracks = self.remove_duplicates(self.tracked_tracks, self.lost_tracks)

        return [t for t in self.tracked_tracks if t.is_activated]

    def _apply_matches(self, matches, tracks, detections, activated, refound):
        for itrack, idet in matches:
            track, det = tracks[itrack], detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated.append(track)
            else:
                track.re_activate(det, self.frame_id)
                refound.append(track)
