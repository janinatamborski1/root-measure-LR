"""Automatic crown-root candidates, to be corrected by hand.

Crown roots start near the top of the primary root. Two cases are found:
  (a) a branch that leaves the primary root's skeleton within CR_ZONE_CM of
      its top and is at least CR_MIN_LEN_CM long;
  (b) a separate skeleton piece (not part of any plant's primary root) that
      starts within CR_NEAR_CM of that zone and is at least CR_MIN_LEN_CM long.
Each is returned as the point where it begins (next to the primary root), so
it can be matched to its plant by nearest primary root.
"""
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

from config import CR_ZONE_CM, CR_NEAR_CM, CR_MIN_LEN_CM, CR_MIN_SEPARATION_CM
from lateral_roots import _neighbor_counts, _walk_branch, _NEIGHBOR_OFFSETS


def _clean_path(path, shape):
    path = np.round(np.asarray(path, dtype=float)).astype(int)
    h, w = shape
    ok = ((path[:, 0] >= 0) & (path[:, 0] < h) &
          (path[:, 1] >= 0) & (path[:, 1] < w))
    return path[ok]


def detect_crown_roots(binary, path, scale, skeleton=None, all_paths=None):
    """Crown-root start points for one plant.

    binary: the plate's root mask. path: Nx2 (row, col) primary root, top to
    tip. scale: pixels per cm. skeleton: optional precomputed skeleton of
    binary. all_paths: primary paths of every plant on the plate, so other
    plants' primary roots are never mistaken for crown roots.
    Returns a list of {'row', 'col'}.
    """
    path = np.asarray(path)
    if path.size == 0:
        return []
    if skeleton is None:
        skeleton = skeletonize(binary)
    h, w = skeleton.shape
    path = _clean_path(path, (h, w))
    if len(path) < 2:
        return []

    # the zone: the first CR_ZONE_CM of the primary root, measured along it
    seg = np.sqrt((np.diff(path.astype(float), axis=0) ** 2).sum(axis=1))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    zone = path[:max(2, int(np.searchsorted(cum, CR_ZONE_CM * scale)) + 1)]

    path_mask = np.zeros((h, w), dtype=bool)
    path_mask[path[:, 0], path[:, 1]] = True
    min_len = max(3, int(CR_MIN_LEN_CM * scale))
    sep = max(3, int(CR_MIN_SEPARATION_CM * scale))
    found = []

    def _is_new(r, c):
        return all(np.hypot(r - fr, c - fc) >= sep for fr, fc in found)

    # (a) branches leaving the primary root inside the zone
    counts = _neighbor_counts(skeleton)
    for r, c in zone:
        if not skeleton[r, c] or counts[r, c] < 3:
            continue
        for dr, dc in _NEIGHBOR_OFFSETS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if not skeleton[nr, nc] or path_mask[nr, nc]:
                continue
            length, _tip = _walk_branch(skeleton, path_mask, (nr, nc))
            if length >= min_len and _is_new(r, c):
                found.append((int(r), int(c)))

    # (b) separate pieces that start near the zone
    labeled, _n = ndimage.label(skeleton, structure=np.ones((3, 3)))
    primary = set(np.unique(labeled[path_mask])) - {0}
    for other in (all_paths or []):
        other = np.asarray(other)
        if other.size == 0:
            continue
        op = _clean_path(other, (h, w))
        if len(op):
            primary |= set(np.unique(labeled[op[:, 0], op[:, 1]])) - {0}
    zone_mask = np.zeros((h, w), dtype=bool)
    zone_mask[zone[:, 0], zone[:, 1]] = True
    near_px = CR_NEAR_CM * scale
    dist = ndimage.distance_transform_edt(~zone_mask)
    for lab, sl in enumerate(ndimage.find_objects(labeled), start=1):
        if sl is None or lab in primary:
            continue
        piece = labeled[sl] == lab
        if piece.sum() < min_len:
            continue
        d = dist[sl][piece]
        k = int(d.argmin())
        if d[k] > near_px:
            continue
        rr, cc = np.nonzero(piece)
        r, c = int(rr[k] + sl[0].start), int(cc[k] + sl[1].start)
        if _is_new(r, c):
            found.append((r, c))

    return [{'row': r, 'col': c} for r, c in found]
