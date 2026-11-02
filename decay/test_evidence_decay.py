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
