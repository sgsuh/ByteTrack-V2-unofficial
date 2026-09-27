"""Kalman filter for 2D bounding boxes in image space.

Adapted from ByteTrack (https://github.com/FoundationVision/ByteTrack, MIT),
which itself follows DeepSORT.

The 8-dimensional state space (u, v, a, h, du, dv, da, dh) contains the box
center (u, v), aspect ratio a = w / h, height h and their velocities. Motion
follows a constant velocity model and (u, v, a, h) is observed directly.
"""
import numpy as np
import scipy.linalg


class KalmanFilter2D:
    ndim = 4

    def __init__(self, dt=1.0):
        self._motion_mat = np.eye(2 * self.ndim)
        for i in range(self.ndim):
            self._motion_mat[i, self.ndim + i] = dt
        self._update_mat = np.eye(self.ndim, 2 * self.ndim)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate (box height).
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement):
        """Create a track (mean, covariance) from an unassociated measurement."""
        mean = np.r_[measurement, np.zeros_like(measurement)]
        h = measurement[3]
        std = [
            2 * self._std_weight_position * h,
            2 * self._std_weight_position * h,
            1e-2,
            2 * self._std_weight_position * h,
            10 * self._std_weight_velocity * h,
            10 * self._std_weight_velocity * h,
            1e-5,
            10 * self._std_weight_velocity * h,
        ]
        return mean, np.diag(np.square(std))

    def _motion_cov(self, h):
        """Process noise Q for an (N,) array of box heights -> (N, 8, 8)."""
        ones = np.ones_like(h)
        std = np.stack([
            self._std_weight_position * h,
            self._std_weight_position * h,
            1e-2 * ones,
            self._std_weight_position * h,
            self._std_weight_velocity * h,
            self._std_weight_velocity * h,
            1e-5 * ones,
            self._std_weight_velocity * h,
        ], axis=1)
        return np.einsum("ni,ij->nij", np.square(std), np.eye(2 * self.ndim))

    def predict(self, mean, covariance):
        mean, covariance = self.multi_predict(mean[None], covariance[None])
        return mean[0], covariance[0]

    def multi_predict(self, mean, covariance):
        """Vectorized prediction step for (N, 8) means and (N, 8, 8) covariances."""
        motion_cov = self._motion_cov(mean[:, 3])
        F = self._motion_mat
        mean = mean @ F.T
        covariance = F @ covariance @ F.T + motion_cov
        return mean, covariance

    def project(self, mean, covariance, score=None, alpha=None):
        """Project the state distribution to measurement space.

        If `score` and `alpha` are given, the measurement noise is scaled as
        R_hat = alpha * (1 - score)^2 * R  (ByteTrackV2, Eq. 10).
        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))
        if score is not None and alpha is not None:
            innovation_cov = innovation_cov * score_noise_scale(score, alpha)
        H = self._update_mat
        return H @ mean, H @ covariance @ H.T + innovation_cov

    def update(self, mean, covariance, measurement, score=None, alpha=None):
        """Measurement-correction step."""
        projected_mean, projected_cov = self.project(mean, covariance, score, alpha)
        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), (covariance @ self._update_mat.T).T, check_finite=False).T
        innovation = measurement - projected_mean
        new_mean = mean + innovation @ kalman_gain.T
        new_covariance = covariance - kalman_gain @ projected_cov @ kalman_gain.T
        return new_mean, new_covariance


def score_noise_scale(score, alpha, min_scale=1e-2):
    """Detection-score based measurement noise factor alpha * (1 - s)^2.

    `min_scale` keeps R positive definite when the score approaches 1.
    """
    return max(alpha * (1.0 - float(score)) ** 2, min_scale)
