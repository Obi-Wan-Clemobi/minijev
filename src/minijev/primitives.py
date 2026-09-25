"""Readouts to typed answers: label probabilities, softmax, the confidence formulas, answer()."""

from __future__ import annotations

import math

import torch


def class_logits(logprobs: torch.Tensor, classes: list[list[int]]) -> tuple[list[float], float]:
    """Combined log-prob per class (logsumexp over label variants), and the total label mass."""
    z = [torch.logsumexp(logprobs[ids], 0).item() for ids in classes]
    mass = sum(math.exp(v) for v in z)
    return z, mass


def softmax(z: list[float], temperature: float = 1.0) -> list[float]:
    m = max(z)
    e = [math.exp((v - m) / temperature) for v in z]
    s = sum(e)
    return [v / s for v in e]


def choice_confidence(p: list[float]) -> float:
    """Peak probability rescaled from uniform (0) to certain (1)."""
    u = 1 / len(p)
    return (max(p) - u) / (1 - u)


def score_confidence(p: list[float]) -> float:
    """Ordinal: 1 − E|level − mode| / E|level − centre| under a uniform distribution, floored at 0.

    Probability on levels far from the mode costs more than on neighbouring levels.
    """
    k = len(p)
    mode = max(range(k), key=p.__getitem__)
    spread = sum(pi * abs(i - mode) for i, pi in enumerate(p))
    uniform_spread = sum(abs(i - (k - 1) / 2) for i in range(k)) / k
    return max(0.0, 1 - spread / uniform_spread)


def answer(q: dict, logits: list[float], temperature: float = 1.0, bias: float = 0.0) -> dict:
    """Uncalibrated at the defaults. bias applies to Nouls only (the Platt shift)."""
    if q["type"] == "noul":
        t = (logits[0] - logits[1]) / temperature + bias
        return {"type": "noul", "noul": 1 / (1 + math.exp(-t)) if t >= 0 else math.exp(t) / (1 + math.exp(t))}
    p = softmax(logits, temperature)
    if q["type"] == "choice":
        keys = list(q["criteria"])
        return {
            "type": "choice",
            "choice": keys[max(range(len(p)), key=p.__getitem__)],
            "probabilities": dict(zip(keys, p)),
            "confidence": choice_confidence(p),
        }
    return {
        "type": "score",
        "score": sum(i * pi for i, pi in enumerate(p)),
        "legend": {str(i): level for i, level in enumerate(q["criteria"])},
        "probabilities": {str(i): pi for i, pi in enumerate(p)},
        "confidence": score_confidence(p),
    }


def rounded(x):
    if isinstance(x, float):
        return round(x, 2)
    if isinstance(x, dict):
        return {k: v if k == "legend" else rounded(v) for k, v in x.items()}
    return x
