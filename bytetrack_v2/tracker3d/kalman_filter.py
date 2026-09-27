"""Constant-velocity Kalman filter for 3D boxes in world coordinates.

State (ByteTrackV2 Sec. 3.2, following AB3DMOT but in the world frame):
    [x, y, z, yaw, l, w, h, vx, vy, vz]
Measurement: [x, y, z, yaw, l, w, h]. Velocities are in m/s and the motion
model is integrated with the actual time step `dt` (seconds) between frames.
"""
import numpy as np

from ..tracker2d.kalman_filter import score_noise_scale

STATE_DIM = 10
MEAS_DIM = 7


def wrap_angle(angle):
    """Wrap angles to [-pi, pi)."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def align_yaw(meas_yaw, ref_yaw):
    """Bring `meas_yaw` within pi/2 of `ref_yaw`, flipping by pi if needed.

    Detectors occasionally predict the heading reversed; a flipped observation
    describes the same box and should not rotate the track by ~180 degrees.
    """
    diff = wrap_angle(meas_yaw - ref_yaw)
    if abs(diff) > np.pi / 2:
        diff = wrap_angle(diff + np.pi)
    return ref_yaw + diff


class KalmanFilter3D:
    """Args:
        init_pos_std: std of the initial box state (x, y, z, yaw, l, w, h).
        init_vel_std: std of the initial velocity.
        proc_pos_std: process noise std of the box state per second.
        proc_vel_std: process noise std of the velocity per second.
        meas_std: measurement noise std R (before score scaling).
    """

    def __init__(self, init_pos_std=1.0, init_vel_std=10.0, proc_pos_std=0.5, proc_vel_std=1.0,
                 meas_std=1.0):
        self.init_cov = np.diag(np.square([init_pos_std] * MEAS_DIM + [init_vel_std] * 3))
        self.proc_std = np.array([proc_pos_std] * MEAS_DIM + [proc_vel_std] * 3)
        self.meas_cov = np.eye(MEAS_DIM) * meas_std ** 2
        self.H = np.eye(MEAS_DIM, STATE_DIM)

    @staticmethod
    def motion_mat(dt):
        F = np.eye(STATE_DIM)
        F[0, 7] = F[1, 8] = F[2, 9] = dt
        return F

    def initiate(self, box, velocity=None):
        """Create a track from a (7,) box and an optional (2,) or (3,) velocity."""
        mean = np.zeros(STATE_DIM)
        mean[:MEAS_DIM] = box
        cov = self.init_cov.copy()
        if velocity is not None:
            velocity = np.asarray(velocity, dtype=np.float64)
            mean[7:7 + len(velocity)] = velocity
        return mean, cov

    def predict(self, mean, covariance, dt):
        F = self.motion_mat(dt)
        Q = np.diag(np.square(self.proc_std * max(dt, 1e-3)))
        mean = F @ mean
        mean[3] = wrap_angle(mean[3])
        return mean, F @ covariance @ F.T + Q

    def update(self, mean, covariance, box, score=None, alpha=None):
        """Correction step. With `score`/`alpha`, R is scaled by alpha * (1 - s)^2 (Eq. 10)."""
        z = np.asarray(box, dtype=np.float64).copy()
        z[3] = align_yaw(z[3], mean[3])
        R = self.meas_cov
        if score is not None and alpha is not None:
            R = R * score_noise_scale(score, alpha)

        H = self.H
        S = H @ covariance @ H.T + R
        K = np.linalg.solve(S, H @ covariance).T
        innovation = z - H @ mean
        new_mean = mean + K @ innovation
        new_mean[3] = wrap_angle(new_mean[3])
        I_KH = np.eye(STATE_DIM) - K @ H
        # Joseph form keeps the covariance symmetric positive definite.
        new_cov = I_KH @ covariance @ I_KH.T + K @ R @ K.T
        return new_mean, new_cov
