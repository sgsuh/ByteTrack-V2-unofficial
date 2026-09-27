import numpy as np

from bytetrack_v2.tracker2d.kalman_filter import KalmanFilter2D, score_noise_scale


def test_constant_velocity_tracking_converges():
    kf = KalmanFilter2D()
    mean, cov = kf.initiate(np.array([100.0, 50.0, 0.5, 80.0]))
    for t in range(1, 30):
        mean, cov = kf.predict(mean, cov)
        mean, cov = kf.update(mean, cov, np.array([100.0 + 3 * t, 50.0, 0.5, 80.0]))
    assert abs(mean[4] - 3.0) < 0.1
    mean, _ = kf.predict(mean, cov)
    assert abs(mean[0] - (100.0 + 3 * 30)) < 1.0


def test_multi_predict_matches_single():
    kf = KalmanFilter2D()
    states = [kf.initiate(np.array([10.0 * i, 5.0, 0.4, 50.0 + i])) for i in range(3)]
    means = np.stack([s[0] for s in states])
    means[:, 4] = 2.0
    covs = np.stack([s[1] for s in states])
    multi_mean, multi_cov = kf.multi_predict(means, covs)
    for i in range(3):
        m, c = kf.predict(means[i], covs[i])
        np.testing.assert_allclose(multi_mean[i], m)
        np.testing.assert_allclose(multi_cov[i], c)


def test_score_adaptive_update_trusts_confident_detections_more():
    kf = KalmanFilter2D()
    mean, cov = kf.initiate(np.array([0.0, 0.0, 0.5, 100.0]))
    mean, cov = kf.predict(mean, cov)
    meas = np.array([10.0, 0.0, 0.5, 100.0])
    high, _ = kf.update(mean, cov, meas, score=0.95, alpha=10)
    low, _ = kf.update(mean, cov, meas, score=0.3, alpha=10)
    assert high[0] > low[0]


def test_score_noise_scale_floor():
    assert score_noise_scale(1.0, 10) > 0
    np.testing.assert_allclose(score_noise_scale(0.5, 100), 25.0)
