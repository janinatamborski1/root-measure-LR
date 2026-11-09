"""Automatic crown-root candidates, to be corrected by hand.

Crown roots grow out of the crown at the base of the shoot, next to the top of
the primary root, and often run beside it for a few centimetres before fanning
out. The skeleton of the plant is treated as a tree rooted at the top of the
primary root. The traced primary root is accepted first. Tips are then taken longest
first and each is followed back until it meets a root that is already
accepted. If that meeting point is within CR_ZONE_CM of the top and the piece
is at least CR_MIN_LEN_CM long, it is a crown root. Long pieces that join
further down are side (lateral) roots and are ignored, as are short stubs.
Each crown root is returned as a point on it, so it can be matched to its plant
by nearest primary root.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from skimage.morphology import skeletonize

from config import CR_ZONE_CM, CR_MIN_LEN_CM, CR_MARK_CM

_ROI_UP_CM = 2.0
_ROI_DOWN_CM = 16.0
_ROI_SIDE_CM = 8.0


def _skeleton_graph(sk):
    ys, xs = np.nonzero(sk)
    n = len(ys)
    idx = -np.ones(sk.shape, dtype=int)
    idx[ys, xs] = np.arange(n)
    I, J, W = [], [], []
    h, w = sk.shape
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        y2, x2 = ys + dy, xs + dx
        ok = (y2 < h) & (x2 >= 0) & (x2 < w)
        a = np.nonzero(ok)[0]
        a = a[sk[y2[a], x2[a]]]
        I += list(a)
        J += list(idx[y2[a], x2[a]])
        W += [float(np.hypot(dy, dx))] * len(a)
    G = coo_matrix((W + W, (I + J, J + I)), shape=(n, n)).tocsr()
    return ys, xs, G


def detect_crown_roots(binary, path, scale, skeleton=None, all_paths=None):
    """Crown-root marker points for one plant.

    binary: the plate's root mask. path: Nx2 (row, col) primary root, top to
    tip. scale: pixels per cm. skeleton: optional precomputed skeleton of
    binary. all_paths: primary paths of every plant on the plate (a tip that
    lies on another plant's primary root is skipped).
    Returns a list of {'row', 'col'}.
    """
    path = np.asarray(path)
    if path.size == 0 or len(path) < 2:
        return []
    if skeleton is None:
        skeleton = skeletonize(binary)
    h, w = skeleton.shape
    r0, c0 = int(round(path[0][0])), int(round(path[0][1]))

    # work on a window around the top of this plant
    ra = max(0, int(r0 - _ROI_UP_CM * scale))
    rb = min(h, int(r0 + _ROI_DOWN_CM * scale))
    ca = max(0, int(c0 - _ROI_SIDE_CM * scale))
    cb = min(w, int(c0 + _ROI_SIDE_CM * scale))
    sk = skeleton[ra:rb, ca:cb]
    if not sk.any():
        return []
    ys, xs, G = _skeleton_graph(sk)
    n = len(ys)
    if n < 2:
        return []

    s = int(np.argmin((ys - (r0 - ra)) ** 2 + (xs - (c0 - ca)) ** 2))
    dist, pred = dijkstra(G, indices=s, return_predecessors=True)
    reach = np.isfinite(dist)
    deg = np.asarray((G > 0).sum(axis=1)).ravel()
    tips = [int(t) for t in np.nonzero(reach & (deg == 1))[0] if t != s]
    tips.sort(key=lambda t: -dist[t])

    # pixels belonging to other plants' primary roots
    other = np.zeros(sk.shape, dtype=bool)
    for p in (all_paths or []):
        p = np.round(np.asarray(p)).astype(int)
        if p.size == 0 or (p[0][0] == path[0][0] and p[0][1] == path[0][1]):
            continue
        rr, cc = p[:, 0] - ra, p[:, 1] - ca
        ok = (rr >= 0) & (rr < sk.shape[0]) & (cc >= 0) & (cc < sk.shape[1])
        other[rr[ok], cc[ok]] = True

    min_len = CR_MIN_LEN_CM * scale
    zone = CR_ZONE_CM * scale
    mark = CR_MARK_CM * scale
    chosen = np.zeros(n, dtype=bool)
    out = []
    # the traced primary root is accepted first, so everything else is
    # compared against it (even if another root happens to be longer)
    node_at = -np.ones(sk.shape, dtype=int)
    node_at[ys, xs] = np.arange(n)
    pp = np.round(path).astype(int)
    rr, cc = pp[:, 0] - ra, pp[:, 1] - ca
    ok = (rr >= 0) & (rr < sk.shape[0]) & (cc >= 0) & (cc < sk.shape[1])
    for r, c in zip(rr[ok], cc[ok]):
        r1, r2, c1, c2 = max(0, r - 2), r + 3, max(0, c - 2), c + 3
        near = node_at[r1:r2, c1:c2]
        chosen[near[near >= 0]] = True
    chosen[s] = True
    first = False
    for t in tips:
        if other[ys[t], xs[t]]:
            continue
        if ys[t] < (r0 - ra) + 1.0 * scale:   # leaves/shoot point up; roots go down
            continue
        walk, v = [], t
        while v >= 0 and not chosen[v] and v != s:
            walk.append(v)
            v = int(pred[v])
        if v < 0 or not walk:
            continue
        piece = dist[t] - dist[v]
        if first:                       # the longest root: the primary root
            chosen[walk] = True
            chosen[v] = True
            first = False
            continue
        if piece < min_len:
            continue
        chosen[walk] = True             # also hides its own side roots
        if dist[v] <= zone:
            # marker a little way along the root, from its base outward
            base = dist[v]
            target = base + min(mark, piece / 2.0)
            m = min(walk, key=lambda q: abs(dist[q] - target))
            out.append({'row': int(ys[m] + ra), 'col': int(xs[m] + ca)})
    return out
