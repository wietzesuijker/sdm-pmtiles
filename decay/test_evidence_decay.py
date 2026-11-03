import numpy as np

from evidence_decay import MAX_KM, apply_curve, distance_km, fit_curve, km_below


def synthetic(curve):
    rng = np.random.default_rng(0)
    clip = np.zeros((200, 200), np.uint8)
    clip[80:120, 80:120] = 1
    ens = rng.integers(0, 100, clip.shape).astype(np.uint8)
    ens[0, :5] = 255
    dist = distance_km(clip)
    return ens, clip, dist, apply_curve(ens, clip, dist, curve)


def test_round_trip_recovers_published_clip():
    true = 1 / (1 + np.arange(MAX_KM + 1) / 30)
    ens, clip, dist, clipped = synthetic(true)
    fitted = fit_curve(ens, clip, clipped, dist)
    err = np.abs(apply_curve(ens, clip, dist, fitted).astype(int) - clipped.astype(int))
    assert np.mean(err <= 1) > 0.99
    assert np.all(np.diff(fitted) <= 0)
    assert (clipped[0, :5] == 255).all()


def test_scale_zero_is_hard_clip_and_inside_untouched():
    ens, clip, dist, _ = synthetic(np.ones(MAX_KM + 1))
    hard = apply_curve(ens, clip, dist, np.ones(MAX_KM + 1), scale=0)
    assert (hard[clip == 1] == ens[clip == 1]).all()
    assert (hard[(clip == 0) & (ens != 255)] == 0).all()


def test_km_below():
    curve = np.array([1, 0.8, 0.4, 0.05])
    assert km_below(curve, 0.5) == 2
    assert km_below(curve, 0.1) == 3
    assert km_below(curve, 0.01) is None


def test_calibration_recovers_true_stretch():
    from calibrate import STRETCHES, loglik

    curve = 1 / (1 + np.arange(MAX_KM + 1) / 30)
    ens, clip, dist, _ = synthetic(curve)
    ens = np.where(ens == 255, 255, 50).astype(np.int16)
    _, logs = loglik(ens, dist, curve, *np.indices(ens.shape).reshape(2, -1), 0.5)
    p = np.exp(logs) / np.exp(logs).sum()
    cells = np.random.default_rng(1).choice(ens.size, 3000, p=p)
    rows, cols = np.unravel_index(cells, ens.shape)
    ll = {}
    for s in STRETCHES:
        total, logs = loglik(ens, dist, curve, rows, cols, s)
        ll[s] = logs.mean() - np.log(total)
    assert max(ll, key=ll.get) == 0.5


def test_uniform_effort_leaves_calibration_unchanged():
    from calibrate import EFFORT_KM, loglik

    curve = 1 / (1 + np.arange(MAX_KM + 1) / 30)
    ens, clip, dist, _ = synthetic(curve)
    rows, cols = np.array([10, 100, 150]), np.array([20, 100, 199])
    effort = np.full((-(-200 // EFFORT_KM),) * 2, 7.0)
    for s in (0, 0.5, 1, float("inf")):
        total, logs = loglik(ens, dist, curve, rows, cols, s)
        total_e, logs_e = loglik(ens, dist, curve, rows, cols, s, effort)
        assert np.allclose(logs - np.log(total), logs_e - np.log(total_e))
