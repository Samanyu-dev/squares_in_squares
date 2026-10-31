"""Enforce a minimum SAT clearance on every non-grid puzzle by a tiny dilation about the box centre (Kaggle's grader needs ~1e-11)."""
import numpy as np, math, sys
sys.path.insert(0, '/Users/apple/Desktop/squares_in_squares')
import build
TARGET = 2e-11
def bbox(P):
    c, s = np.cos(P[:, 2]), np.sin(P[:, 2]); h = .5 * (abs(c) + abs(s))
    return max((P[:, 0] + h).max() - (P[:, 0] - h).min(), (P[:, 1] + h).max() - (P[:, 1] - h).min())
def enforce(n, P, target=TARGET):
    """returns (P2, gap, delta_used). Grid-valued puzzles are returned unchanged."""
    P = np.array(P, float).copy(); L = bbox(P)
    if n < 2 or abs(L - math.ceil(math.sqrt(n))) < 1e-9: return P, None, 0.0
    delta = 0.0
    for _ in range(8):
        g = build.min_gap(P)
        if g >= target: return P, g, delta
        d = (target - g) * 1.6 + 1e-13; c = bbox(P) / 2
        P[:, 0] = c + (P[:, 0] - c) * (1 + d); P[:, 1] = c + (P[:, 1] - c) * (1 + d); delta += d
    return P, build.min_gap(P), delta
