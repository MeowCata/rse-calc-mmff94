"""
Geometric strain analysis from MMFF94-optimized geometry.

Computes individual strain components:
- Baeyer (angle) strain: deviation of bond angles from ideal values
- Pitzer (torsional) strain: eclipsing interactions from torsion angles
- Transannular strain: close non-bonded contacts in medium/large rings

Provides qualitative decomposition of where ring strain originates.
"""

from typing import Dict, List, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import Mol


# ---------------------------------------------------------------------------
# Ideal angles by hybridization
# ---------------------------------------------------------------------------

IDEAL_ANGLES = {
    "sp3": 109.5,
    "sp2": 120.0,
    "sp": 180.0,
    "aromatic": 120.0,
}

# Preferred torsion angles for saturated rings (degrees)
IDEAL_TORSIONS = {
    "sp3-sp3_sp3-sp3": (60., -60., 180.),
    "sp3-sp3_sp3-sp2": (60., -60., 180.),
    "sp2-sp3_sp3-sp2": (60., -60., 180.),
}


class GeometryAnalyzer:
    """Compute geometric strain components from a 3D conformer.

    Parameters
    ----------
    mol : Mol
        RDKit molecule with a 3D conformer (from MMFF94 optimization).
    conf_id : int
        Which conformer to analyze (default: -1 = last).
    """

    def __init__(self, mol: Mol, conf_id: int = -1):
        self.mol = mol
        self.conf = mol.GetConformer(conf_id)
        if not self.conf.Is3D():
            raise ValueError("Molecule must have a 3D conformer.")
        # Cache coordinates once; every diagnostic reads the same immutable
        # optimized geometry and should not rebuild small arrays per atom.
        self._positions = np.asarray(self.conf.GetPositions(), dtype=float)

    # ------------------------------------------------------------------
    # Bond angle (Baeyer) strain
    # ------------------------------------------------------------------

    def compute_bond_angle(self, i: int, j: int, k: int) -> float:
        """Compute bond angle at atom j (degrees) for atoms i-j-k."""
        pos_i = self._pos(i)
        pos_j = self._pos(j)
        pos_k = self._pos(k)

        v1 = pos_i - pos_j
        v2 = pos_k - pos_j

        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_angle)))

    def get_ideal_angle(self, atom_idx: int) -> float:
        """Return the ideal bond angle for an atom based on its hybridization."""
        atom = self.mol.GetAtomWithIdx(atom_idx)
        hyb = atom.GetHybridization()
        if hyb == Chem.HybridizationType.SP:
            return 120.0
        elif hyb == Chem.HybridizationType.SP2:
            return 120.0
        elif hyb == Chem.HybridizationType.SP3:
            return 109.5
        elif hyb == Chem.HybridizationType.SP3D:
            return 109.5
        else:
            return 109.5  # Default for saturated atoms

    def compute_angle_strain_in_ring(
        self,
        ring_atoms: Tuple[int, ...],
    ) -> Dict:
        """Compute Baeyer (angle) strain for a ring.

        Returns dict with mean deviation, max deviation, and list of
        strained angles (>5 degrees from ideal).
        """
        angles = []
        deviations = []
        n = len(ring_atoms)

        for idx in range(n):
            i = ring_atoms[(idx - 1) % n]
            j = ring_atoms[idx]
            k = ring_atoms[(idx + 1) % n]

            actual = self.compute_bond_angle(i, j, k)
            ideal = self.get_ideal_angle(j)
            dev = actual - ideal

            angles.append({
                "atom": j,
                "symbol": self.mol.GetAtomWithIdx(j).GetSymbol(),
                "actual": round(actual, 2),
                "ideal": ideal,
                "deviation": round(dev, 2),
            })
            deviations.append(dev)

        strained = [a for a in angles if abs(a["deviation"]) > 5.0]

        return {
            "angles": angles,
            "mean_deviation": round(float(np.mean(deviations)), 2),
            "max_deviation": round(max(abs(d) for d in deviations), 2),
            "rms_deviation": round(float(np.sqrt(np.mean(np.array(deviations)**2))), 2),
            "strained_atoms": strained,
            "n_strained": len(strained),
        }

    # ------------------------------------------------------------------
    # Torsion (Pitzer) strain
    # ------------------------------------------------------------------

    def compute_torsion_angle(
        self, i: int, j: int, k: int, l: int
    ) -> float:
        """Compute a signed torsion angle, or NaN for degenerate geometry."""
        p0, p1, p2, p3 = self._positions[[i, j, k, l]]
        b0 = p0 - p1
        axis = p2 - p1
        b2 = p3 - p2
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-10:
            return float("nan")
        axis /= axis_norm

        v = b0 - np.dot(b0, axis) * axis
        w = b2 - np.dot(b2, axis) * axis
        v_norm = np.linalg.norm(v)
        w_norm = np.linalg.norm(w)
        if v_norm < 1e-10 or w_norm < 1e-10:
            return float("nan")
        v /= v_norm
        w /= w_norm

        x = float(np.clip(np.dot(v, w), -1.0, 1.0))
        y = float(np.dot(np.cross(axis, v), w))
        return float(np.degrees(np.arctan2(y, x)))

    def compute_torsion_strain_in_ring(
        self,
        ring_atoms: Tuple[int, ...],
    ) -> Dict:
        """Compute Pitzer (torsional) strain for a ring.

        Each ring bond is viewed down its bond axis. The closest projected
        neighbour pair defines its eclipse deviation; a bond is counted once
        when that deviation is at most 20 degrees. This handles all explicit
        substituents and avoids the invalid A-B-C-A pseudo-dihedrals that a
        three-membered ring produces with a backbone-only calculation.
        """
        torsions = []
        ring_set = set(ring_atoms)
        ring_bonds = sorted(
            (
                bond for bond in self.mol.GetBonds()
                if bond.GetBeginAtomIdx() in ring_set
                and bond.GetEndAtomIdx() in ring_set
            ),
            key=lambda bond: bond.GetIdx(),
        )

        for bond in ring_bonds:
            j = bond.GetBeginAtomIdx()
            k = bond.GetEndAtomIdx()
            axis = self._pos(k) - self._pos(j)
            axis_norm = np.linalg.norm(axis)
            if axis_norm < 1e-10:
                continue
            axis /= axis_norm

            left = [
                atom.GetIdx()
                for atom in self.mol.GetAtomWithIdx(j).GetNeighbors()
                if atom.GetIdx() != k
            ]
            right = [
                atom.GetIdx()
                for atom in self.mol.GetAtomWithIdx(k).GetNeighbors()
                if atom.GetIdx() != j
            ]

            closest = None
            for i in left:
                v = self._pos(i) - self._pos(j)
                v -= np.dot(v, axis) * axis
                v_norm = np.linalg.norm(v)
                if v_norm < 1e-10:
                    continue
                v /= v_norm

                for l in right:
                    if i == l:
                        continue
                    w = self._pos(l) - self._pos(k)
                    w -= np.dot(w, axis) * axis
                    w_norm = np.linalg.norm(w)
                    if w_norm < 1e-10:
                        continue
                    w /= w_norm

                    x = float(np.clip(np.dot(v, w), -1.0, 1.0))
                    y = float(np.dot(axis, np.cross(v, w)))
                    angle = float(np.degrees(np.arctan2(y, x)))
                    deviation = abs(angle)
                    if closest is None or deviation < closest[0]:
                        closest = (deviation, angle, i, l)

            if closest is None:
                torsions.append({
                    "bond": (j, k),
                    "atoms": None,
                    "angle": None,
                    "eclipse_deviation": None,
                    "category": "undefined",
                })
                continue

            deviation, angle, i, l = closest
            if deviation <= 20.0:
                category = "eclipsed"
            elif deviation >= 40.0:
                category = "staggered"
            else:
                category = "intermediate"
            torsions.append({
                "bond": (j, k),
                "atoms": (i, j, k, l),
                "angle": round(angle, 2),
                "eclipse_deviation": round(deviation, 2),
                "category": category,
            })

        eclipsed = [t for t in torsions if t["category"] == "eclipsed"]

        return {
            "torsions": torsions,
            "n_eclipsed": len(eclipsed),
        }

    # ------------------------------------------------------------------
    # Transannular strain (non-bonded contacts)
    # ------------------------------------------------------------------

    def compute_transannular_contacts(
        self,
        ring_atoms: Tuple[int, ...],
        threshold: float = 2.5,
        skip_bonded: bool = True,
    ) -> List[Dict]:
        """Find close non-bonded contacts in a ring (transannular strain).

        Args:
            ring_atoms: Indices of atoms in the ring.
            threshold: Distance cutoff for reporting contacts (angstroms).
            skip_bonded: Ignore 1,2- and 1,3-bonded pairs.

        Returns:
            List of dicts with atom pairs and distances.
        """
        contacts = []
        n = len(ring_atoms)

        for i in range(n):
            for j in range(i + 1, n):
                a1 = ring_atoms[i]
                a2 = ring_atoms[j]

                # Skip adjacent atoms and 1-3 neighbors
                if skip_bonded:
                    dist_in_ring = min(abs(i - j), n - abs(i - j))
                    if dist_in_ring <= 2:
                        continue

                d = self.distance(a1, a2)
                if d < threshold:
                    contacts.append({
                        "atom1": a1,
                        "atom2": a2,
                        "symbol1": self.mol.GetAtomWithIdx(a1).GetSymbol(),
                        "symbol2": self.mol.GetAtomWithIdx(a2).GetSymbol(),
                        "distance": round(d, 2),
                    })

        return contacts

    # ------------------------------------------------------------------
    # Full ring analysis
    # ------------------------------------------------------------------

    def analyze_ring(
        self,
        ring_atoms: Tuple[int, ...],
    ) -> Dict:
        """Comprehensive geometric strain analysis for a single ring.

        Returns a dict with angle strain, torsional strain, and
        transannular contact information.
        """
        size = len(ring_atoms)

        return {
            "ring_size": size,
            "atoms": list(ring_atoms),
            "angle_strain": self.compute_angle_strain_in_ring(ring_atoms),
            "torsional_strain": self.compute_torsion_strain_in_ring(ring_atoms),
            "transannular_contacts": self.compute_transannular_contacts(ring_atoms),
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def distance(self, i: int, j: int) -> float:
        """Compute Euclidean distance between two atoms (angstroms)."""
        pi = self._pos(i)
        pj = self._pos(j)
        return float(np.linalg.norm(pi - pj))

    def _pos(self, idx: int) -> np.ndarray:
        """Get 3D position of an atom as numpy array."""
        return self._positions[idx]
