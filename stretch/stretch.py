"""Stretch a packing by whole arm columns / rows (Squares in Squares).

A packing with tilted arms along the walls is cut with one vertical line (through a horizontal arm and the upright
block) and one horizontal line (through a vertical arm and the block). Insert (+1): copy the arm stack at the cut and
one upright column / row of the block next to it, and shift everything beyond the cut by the arm's measured pitch.
Delete (-1): remove them; arm squares beyond the cut move back by the pitch, everything else by 1. K -> K +- 1 per
(x, y) pair of operations; fq then relaxes the result.
Packings are arrays of (x, y, angle in radians).
"""
import math

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def patches(a):
    """Connected groups (>= 15 squares) of tilted squares, with their lattice indices along / across the tilt."""
    deg = np.degrees(np.mod(a[:, 2], math.pi / 2)); deg = np.where(deg > 45, deg - 90, deg)
    t = np.where(np.abs(deg) > 3)[0]
    res = []
    if len(t) == 0:
        return res
    P = a[t, :2]
    pr = cKDTree(P).query_pairs(1.6, output_type='ndarray')
    A = coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(len(P), len(P))) if len(pr) else coo_matrix((len(P), len(P)))
    nc, lab = connected_components(A, directed=False)
    for k in range(nc):
        idx = t[lab == k]
        if len(idx) < 15:
            continue
        th = math.radians(float(np.median(deg[idx])))
        u = np.array([math.cos(th), math.sin(th)]); v = np.array([-math.sin(th), math.cos(th)])
        pu, pv = a[idx, :2] @ u, a[idx, :2] @ v
        iu = np.round(pu - np.mean(pu - np.round(pu))).astype(int)
        iv = np.round(pv - np.mean(pv - np.round(pv))).astype(int)
        horiz = np.ptp(a[idx, 0]) >= np.ptp(a[idx, 1])
        res.append(dict(idx=idx, th=th, iu=iu, iv=iv, horiz=horiz,
                        x0=a[idx, 0].min(), x1=a[idx, 0].max(), y0=a[idx, 1].min(), y1=a[idx, 1].max()))
    return res


def stack_info(a, p, axis):
    """Stack index of every patch square across the cut, and the pitch: median centre distance between neighbouring
    stacks along the axis, measured on the middle half of the patch."""
    k = p['iu'] if axis == 0 else p['iv']
    coord = a[p['idx'], axis]
    ks = np.unique(k)
    means = np.array([coord[k == q].mean() for q in ks])
    mid = slice(len(ks) // 4, max(len(ks) // 4 + 2, 3 * len(ks) // 4))
    d = np.diff(means)[mid]
    pitch = float(np.median(np.abs(d))) if len(d) else 1.0
    return k, ks, means, pitch


def apply(a, axis, cut, op):
    """One insert (+1) or delete (-1) across `axis` (0: vertical cut at x = cut, 1: horizontal cut at y = cut)."""
    P = patches(a)
    cross = [p for p in P if (p['x0'] < cut < p['x1'] if axis == 0 else p['y0'] < cut < p['y1'])]
    good = [p for p in cross if p['horiz'] == (axis == 0)]
    if len(cross) != len(good) or not good:
        return None, 'cut crosses a patch along its length' if cross else 'no patch at the cut'
    in_patch = np.zeros(len(a), bool)
    beyond = a[:, axis] > cut
    copy_mask = np.zeros(len(a), bool)
    shift = np.zeros(len(a))
    pitches = []
    for p in good:
        k, ks, means, pitch = stack_info(a, p, axis)
        pitches.append(pitch)
        kc = ks[np.argmin(np.abs(means - cut))]
        in_patch[p['idx']] = True
        beyond[p['idx']] = k > kc
        copy_mask[p['idx'][k == kc]] = True
        if op < 0:
            shift[p['idx'][k > kc]] = -pitch
    d = max([1.0] + pitches)
    up_col = (~in_patch) & (a[:, axis] > cut - 1.0) & (a[:, axis] <= cut)
    deg = np.degrees(np.mod(a[:, 2], math.pi / 2)); upright = np.minimum(deg, 90 - deg) < 3
    up_col &= upright
    b = a.copy()
    if op > 0:
        b[beyond, axis] += d
        cp = a[copy_mask | up_col].copy()
        cp[:, axis] += d
        b = np.vstack([b, cp])
    else:
        rest = beyond & ~in_patch
        shift[rest] = -1.0
        b[:, axis] += shift
        b = b[~(copy_mask | up_col)]
    return b, f'axis {axis} cut {cut:.2f} op {op:+d} pitch {d:.4f} copied/removed {int((copy_mask | up_col).sum())}'


def pick_cuts(a, axis):
    """Cut through the middle of the largest horizontal (axis 0) / vertical (axis 1) arm, on an integer line of the
    upright block, without cutting another patch along its length."""
    P = [p for p in patches(a) if p['horiz'] == (axis == 0)]
    if not P:
        return None
    p = max(P, key=lambda q: len(q['idx']))
    lo, hi = (p['x0'], p['x1']) if axis == 0 else (p['y0'], p['y1'])
    c = math.floor((lo + hi) / 2) + 0.0
    for dc in (0, 1, -1, 2, -2, 3, -3):
        cc = c + dc
        others = [q for q in patches(a) if q['horiz'] != (axis == 0) and
                  ((q['x0'] - 0.8 < cc < q['x1'] + 0.8) if axis == 0 else (q['y0'] - 0.8 < cc < q['y1'] + 0.8))]
        if not others and lo + 2 < cc < hi - 2:
            return cc
    return None


def stretch(a, steps):
    """K -> K + steps (steps < 0 deletes). Returns (packing, notes) or (None, reason)."""
    b, notes = np.array(a, float), []
    for _ in range(abs(steps)):
        for axis in (0, 1):
            c = pick_cuts(b, axis)
            if c is None:
                return None, f'no cut on axis {axis}'
            b2, note = apply(b, axis, c, 1 if steps > 0 else -1)
            if b2 is None:
                return None, note
            b = b2; notes.append(note)
    return b, notes