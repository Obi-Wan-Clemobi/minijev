"""Calibration fits: temperature and Platt for yes/no log-odds, temperature for multi-class logits."""

from __future__ import annotations

import math


def sigmoid(t: float) -> float:
    return 1 / (1 + math.exp(-t)) if t >= 0 else math.exp(t) / (1 + math.exp(t))


def softplus(t: float) -> float:
    return t + math.log1p(math.exp(-t)) if t > 0 else math.log1p(math.exp(t))


def nll(z: list[float], y: list[int], a: float, b: float) -> float:
    return sum(softplus(-(a * zi + b)) if yi else softplus(a * zi + b) for zi, yi in zip(z, y)) / len(z)


def fit_affine(z: list[float], y: list[int], slope_only: bool) -> tuple[float, float]:
    """Minimize NLL of sigmoid(a*z + b) (Platt) or sigmoid(z / T) with T = 1/a (temperature).

    Newton steps with backtracking: saturated sigmoids have almost no curvature, so a full
    Newton step can overshoot wildly. Only accept a step that lowers the NLL.
    """
    a, b = 1.0, 0.0
    for _ in range(100):
        ga = gb = haa = hab = hbb = 0.0
        for zi, yi in zip(z, y):
            s = sigmoid(a * zi + b)
            r, w = s - yi, s * (1 - s)
            ga, gb = ga + r * zi, gb + r
            haa, hab, hbb = haa + w * zi * zi, hab + w * zi, hbb + w
        if slope_only:
            da, db = ga / (haa + 1e-9), 0.0
        else:
            det = haa * hbb - hab * hab + 1e-9
            da, db = (hbb * ga - hab * gb) / det, (haa * gb - hab * ga) / det
        current, step = nll(z, y, a, b), 1.0
        while step > 1e-6 and nll(z, y, a - step * da, b - step * db) > current:
            step /= 2
        if step <= 1e-6:
            break
        a, b = a - step * da, b - step * db
    return a, b


def cross_fit(z: list[float], y: list[int], slope_only: bool) -> list[float]:
    """Two-fold: fit on one half, apply to the other. Returns out-of-fold calibrated log-odds."""
    half = len(z) // 2
    out = [0.0] * len(z)
    for fit_idx, apply_idx in ((range(half, len(z)), range(half)), (range(half), range(half, len(z)))):
        a, b = fit_affine([z[i] for i in fit_idx], [y[i] for i in fit_idx], slope_only)
        for i in apply_idx:
            out[i] = a * z[i] + b
    return out


def logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def fit_temperature_multiclass(logits: list[list[float]], labels: list[int]) -> float:
    """T that minimizes the NLL of softmax(z / T), by golden-section search on log T."""
    def nll_t(log_t: float) -> float:
        t = math.exp(log_t)
        total = 0.0
        for z, y in zip(logits, labels):
            m = max(v / t for v in z)
            total -= z[y] / t - m - math.log(sum(math.exp(v / t - m) for v in z))
        return total / len(labels)
    a, b, g = -2.0, 3.0, (math.sqrt(5) - 1) / 2
    for _ in range(60):
        c, d = b - g * (b - a), a + g * (b - a)
        a, b = (a, d) if nll_t(c) < nll_t(d) else (c, b)
    return math.exp((a + b) / 2)


def softmax_list(z: list[float]) -> list[float]:
    m = max(z)
    e = [math.exp(v - m) for v in z]
    return [v / sum(e) for v in e]


def cross_fit_multiclass(logits: list[list[float]], labels: list[int]) -> list[list[float]]:
    """Two-fold temperature scaling: fit T on one half, apply it to the other. Out-of-fold probabilities."""
    half, out = len(logits) // 2, [None] * len(logits)
    for fit_idx, apply_idx in ((range(half, len(logits)), range(half)), (range(half), range(half, len(logits)))):
        t = fit_temperature_multiclass([logits[i] for i in fit_idx], [labels[i] for i in fit_idx])
        for i in apply_idx:
            out[i] = [v / t for v in logits[i]]
    return [softmax_list(z) for z in out]


def nll_multi(logits: list[list[float]], y: list[int], t: float = 1.0) -> float:
    total = 0.0
    for z, yi in zip(logits, y):
        m = max(v / t for v in z)
        total -= z[yi] / t - m - math.log(sum(math.exp(v / t - m) for v in z))
    return total / len(y)
