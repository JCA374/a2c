"""The lidar ray cast. `cast_python` is the reference; `cast_numba` is the same source compiled with numba
(chosen by the bake-off in experiments/bakeoff, bit-identical on 900 poses x 19 rays; tests/test_lidar.py)."""
from __future__ import annotations

import numpy as np


def _inside(tiles, t, px, py):
    """Crossing-number point-in-polygon test (the v1 `_point_in_poly`) for quad t of tiles[T*8]."""
    b = t * 8
    inside = False
    p1x, p1y = tiles[b], tiles[b + 1]
    for i in range(5):
        j = b + (i % 4) * 2
        p2x, p2y = tiles[j], tiles[j + 1]
        if py > min(p1y, p2y):
            if py <= max(p1y, p2y):
                if px <= max(p1x, p2x):
                    if p1y != p2y:
                        xin = (py - p1y) * (p2x - p1x) / (p2y - p1y + 1e-8) + p1x
                    else:
                        xin = p1x
                    if p1x == p2x or px <= xin:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def _cast(tiles, centers, T, x, y, dirs, R, step, maxd, out, cand):
    """Distance (0..1 of maxd) along each ray to the first sample that is on no road tile."""
    lim = (maxd + 10.0) * (maxd + 10.0)
    nc = 0
    for t in range(T):
        dx, dy = centers[2 * t] - x, centers[2 * t + 1] - y
        if dx * dx + dy * dy <= lim:
            cand[nc] = t
            nc += 1
    K = int(maxd / step)
    last = 0
    for r in range(R):
        dx, dy = dirs[2 * r], dirs[2 * r + 1]
        res = 1.0
        for k in range(1, K + 1):
            px = x + dx * (step * k)
            py = y + dy * (step * k)
            hit = False
            if nc > 0 and _inside(tiles, cand[last], px, py):
                hit = True
            else:
                for c in range(nc):
                    if _inside(tiles, cand[c], px, py):
                        hit = True
                        last = c
                        break
            if not hit:
                res = step * (k - 1) / maxd
                break
        out[r] = res


def cast_python(tiles, centers, x, y, dirs, step, maxd):
    R = len(dirs) // 2
    out = [0.0] * R
    T = len(centers) // 2
    _cast(tiles.tolist(), centers.tolist(), T, float(x), float(y), dirs.tolist(), R, step, maxd, out, [0] * T)
    return np.array(out)


_numba_cast = None


def cast_numba(tiles, centers, x, y, dirs, step, maxd):
    global _numba_cast
    if _numba_cast is None:
        from numba import njit

        inside = njit(cache=True, nogil=True, inline="always")(_inside)
        g = dict(_cast.__globals__, _inside=inside)
        import types

        _numba_cast = njit(cache=True, nogil=True)(types.FunctionType(_cast.__code__, g, "_cast_nb"))
    R = len(dirs) // 2
    T = len(centers) // 2
    out = np.empty(R)
    _numba_cast(tiles, centers, T, float(x), float(y), dirs, R, float(step), float(maxd), out, np.empty(T, np.int64))
    return out


CASTERS = {"python": cast_python, "numba": cast_numba}
