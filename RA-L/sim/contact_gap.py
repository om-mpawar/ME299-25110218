"""
Smallest gap left between facing disc faces: does a pose need disc contact at all?

Without friction, contact can only push. If the equilibrium found WITHOUT contact keeps
every notch gap open, no contact force would act, so it is also the equilibrium WITH
contact (the unconstrained energy minimum already satisfies the constraints). Only the
poses that close a gap have to be simulated with SOFA's (expensive) contact detection.

The gap is measured on the facing node pairs stored in data/contacts/<mesh>.contact.npz
by tools/make_contact_surface.py: node i on one face of a notch, node j on the face across
it. Each node's distance from the tangent plane of the other face (outward normals from
the current contact triangles) is computed and the smaller one kept. Negative: the faces
have passed through each other.

    gc = GapCheck("data/contacts/one_section_cad_sym.contact.npz")
    gc.min_gap(positions)            # mm
"""
import os

import numpy as np

from tdcr.paths import ROOT


def contact_file(mesh):
    return os.path.join(ROOT, "data", "contacts", os.path.basename(mesh)[:-4] + ".contact.npz")


class GapCheck:
    """mode "normal" (default): distance of each node from the other face's tangent plane.
    mode "rotated" (TDCR_GAP_MODE=rotated): the pair's rest separation vector, carried along
    with the local rotation of the material around node i (Kabsch on its 12 nearest nodes),
    projected on the current separation. Exact at rest; robust on curved faces (the rolling-disc
    notches of the standard module, where vertex normals mix notch and channel walls)."""

    def __init__(self, path, rest=None, mode=None):
        d = np.load(path)
        if "pair_i" not in d.files:
            raise ValueError(f"{path} has no facing pairs: re-run tools/make_contact_surface.py")
        self.tri = d["triangles"]
        self.i, self.j = d["pair_i"], d["pair_j"]
        self.mode = mode or os.environ.get("TDCR_GAP_MODE", "normal")
        if self.mode == "rotated":
            if rest is None:
                raise ValueError("GapCheck(mode='rotated') needs the rest positions")
            from scipy.spatial import cKDTree
            self.rest = np.asarray(rest, dtype=float)
            v = self.rest[self.j] - self.rest[self.i]
            self.n0 = v / np.linalg.norm(v, axis=1, keepdims=True)
            self.nb = cKDTree(self.rest).query(self.rest[self.i], k=12)[1]

    def _rotated(self, pos):
        A = self.rest[self.nb] - self.rest[self.nb].mean(1, keepdims=True)
        B = pos[self.nb] - pos[self.nb].mean(1, keepdims=True)
        U, S, Vt = np.linalg.svd(np.einsum("pki,pkj->pij", A, B))
        dsg = np.sign(np.linalg.det(np.einsum("pij,pjk->pik", U, Vt)))
        D = np.zeros_like(U); D[:, 0, 0] = 1; D[:, 1, 1] = 1; D[:, 2, 2] = dsg
        R = np.einsum("pij,pjk,pkl->pli", U, D, Vt)                 # rest -> current rotation
        return np.einsum("pi,pi->p", pos[self.j] - pos[self.i], np.einsum("pij,pj->pi", R, self.n0))

    def normals(self, pos):
        """Outward unit normal at every node of the contact triangles (area weighted)."""
        a, b, c = (pos[self.tri[:, k]] for k in range(3))
        fn = np.cross(b - a, c - a)                  # triangles are wound outward
        vn = np.zeros_like(pos)
        for k in range(3):
            np.add.at(vn, self.tri[:, k], fn)
        return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-30)

    def gaps(self, pos):
        pos = np.asarray(pos, dtype=float)
        if self.mode == "rotated":
            return self._rotated(pos)
        n = self.normals(pos)
        d = pos[self.j] - pos[self.i]
        return np.minimum(np.einsum("ij,ij->i", d, n[self.i]), np.einsum("ij,ij->i", -d, n[self.j]))

    def min_gap(self, pos):
        return float(self.gaps(pos).min())
