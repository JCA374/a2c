"""SPEC.md in pure Python (the reference), numba and NumPy."""
import numpy as np


def _inside(tiles, t, px, py):
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


def _cast(tiles, centers, T, pos, dirs, p, R, step, maxd, out, cand):
    x, y = pos[2 * p], pos[2 * p + 1]
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
        q = (p * R + r) * 2
        dx, dy = dirs[q], dirs[q + 1]
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
        out[p * R + r] = res


def lidar_ref(tiles, centers, T, pos, dirs, P, R, step, maxd):
    tl, cl, pl, dl = tiles.tolist(), centers.tolist(), pos.tolist(), dirs.tolist()
    out = [0.0] * (P * R)
    cand = [0] * T
    for p in range(P):
        _cast(tl, cl, T, pl, dl, p, R, step, maxd, out, cand)
    return np.array(out)


def make_numba():
    from numba import njit, prange
    inside = njit(cache=True, nogil=True, inline='always')(_inside)
    cast_src = njit(cache=True, nogil=True)

    @cast_src
    def cast(tiles, centers, T, pos, dirs, p, R, step, maxd, out, cand):
        x, y = pos[2 * p], pos[2 * p + 1]
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
            q = (p * R + r) * 2
            dx, dy = dirs[q], dirs[q + 1]
            res = 1.0
            for k in range(1, K + 1):
                px = x + dx * (step * k)
                py = y + dy * (step * k)
                hit = False
                if nc > 0 and inside(tiles, cand[last], px, py):
                    hit = True
                else:
                    for c in range(nc):
                        if inside(tiles, cand[c], px, py):
                            hit = True
                            last = c
                            break
                if not hit:
                    res = step * (k - 1) / maxd
                    break
            out[p * R + r] = res

    @njit(cache=True, nogil=True)
    def serial(tiles, centers, T, pos, dirs, P, R, step, maxd, out):
        cand = np.empty(T, np.int64)
        for p in range(P):
            cast(tiles, centers, T, pos, dirs, p, R, step, maxd, out, cand)

    @njit(cache=True, nogil=True, parallel=True)
    def parallel(tiles, centers, T, pos, dirs, P, R, step, maxd, out):
        for p in prange(P):
            cand = np.empty(T, np.int64)
            cast(tiles, centers, T, pos, dirs, p, R, step, maxd, out, cand)

    return serial, parallel


def lidar_numpy(tiles, centers, T, pos, dirs, P, R, step, maxd):
    """All samples of all rays at once (no early exit), the same crossing-number test, vectorised."""
    v = tiles.reshape(T, 4, 2)
    K = int(maxd / step)
    out = np.empty((P, R))
    pos2, dirs3 = pos.reshape(P, 2), dirs.reshape(P, R, 2)
    ks = step * np.arange(1, K + 1)
    lim = (maxd + 10.0) * (maxd + 10.0)
    for p in range(P):
        x, y = pos2[p]
        d = centers.reshape(T, 2) - pos2[p]
        cand = v[(d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1]) <= lim]           # C x 4 x 2
        px = x + dirs3[p, :, 0:1] * ks[None]                               # R x K
        py = y + dirs3[p, :, 1:2] * ks[None]
        px, py = px[..., None], py[..., None]                              # R x K x 1
        inside = np.zeros(px.shape[:2] + (len(cand),), bool)
        for i in range(5):
            p1, p2 = cand[:, (i - 1) % 4] if i else cand[:, 0], cand[:, i % 4]
            p1x, p1y, p2x, p2y = p1[:, 0], p1[:, 1], p2[:, 0], p2[:, 1]
            if i == 0:
                continue                                                   # edge v0 -> v0 never crosses
            cond = (py > np.minimum(p1y, p2y)) & (py <= np.maximum(p1y, p2y)) & (px <= np.maximum(p1x, p2x))
            with np.errstate(divide='ignore', invalid='ignore'):
                xin = np.where(p1y != p2y, (py - p1y) * (p2x - p1x) / (p2y - p1y + 1e-8) + p1x, p1x)
            inside ^= cond & ((p1x == p2x) | (px <= xin))
        on = inside.any(-1)                                                # R x K
        first_off = np.where(on.all(1), K + 1, np.argmin(on, 1) + 1)       # 1-based sample index
        out[p] = np.where(first_off > K, 1.0, step * (first_off - 1) / maxd)
    return out.reshape(-1)
