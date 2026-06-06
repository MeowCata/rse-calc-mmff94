"""
Reference compound database for ring strain energy calibration.

Contains experimentally known ring strain values from the literature
and provides lookup, calibration data export, and validation utilities.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Reference compound data
# ---------------------------------------------------------------------------

@dataclass
class ReferenceCompound:
    name: str
    smiles: str
    ring_sizes: List[int]
    ring_count: int
    strain_energy_kcal_mol: float
    source: str
    method: str = "combustion calorimetry"
    notes: str = ""
    per_ring_strain: Optional[Dict[int, float]] = None


# ---------------------------------------------------------------------------
# Built-in reference dataset
# ---------------------------------------------------------------------------

_REFERENCE_COMPOUNDS: List[ReferenceCompound] = [
    # ---------- Monocyclic saturated hydrocarbons ----------
    ReferenceCompound(
        name="cyclopropane",
        smiles="C1CC1",
        ring_sizes=[3],
        ring_count=1,
        strain_energy_kcal_mol=27.5,
        source="Wiberg (1986), Angew. Chem. Int. Ed. 25, 312",
        method="combustion calorimetry",
    ),
    ReferenceCompound(
        name="cyclobutane",
        smiles="C1CCC1",
        ring_sizes=[4],
        ring_count=1,
        strain_energy_kcal_mol=26.3,
        source="Wiberg (1986), Angew. Chem. Int. Ed. 25, 312",
        method="combustion calorimetry",
    ),
    ReferenceCompound(
        name="cyclopentane",
        smiles="C1CCCC1",
        ring_sizes=[5],
        ring_count=1,
        strain_energy_kcal_mol=6.5,
        source="Wiberg (1986), Angew. Chem. Int. Ed. 25, 312",
        method="combustion calorimetry",
    ),
    ReferenceCompound(
        name="cyclohexane",
        smiles="C1CCCCC1",
        ring_sizes=[6],
        ring_count=1,
        strain_energy_kcal_mol=0.0,
        source="Wiberg (1986), Angew. Chem. Int. Ed. 25, 312",
        method="combustion calorimetry",
        notes="Strain-free reference ring size.",
    ),
    ReferenceCompound(
        name="cycloheptane",
        smiles="C1CCCCCC1",
        ring_sizes=[7],
        ring_count=1,
        strain_energy_kcal_mol=6.3,
        source="Wiberg (1986), Angew. Chem. Int. Ed. 25, 312",
        method="combustion calorimetry",
    ),
    ReferenceCompound(
        name="cyclooctane",
        smiles="C1CCCCCCC1",
        ring_sizes=[8],
        ring_count=1,
        strain_energy_kcal_mol=9.7,
        source="Wiberg (1986), Angew. Chem. Int. Ed. 25, 312",
        method="combustion calorimetry",
    ),
    ReferenceCompound(
        name="cyclononane",
        smiles="C1CCCCCCCC1",
        ring_sizes=[9],
        ring_count=1,
        strain_energy_kcal_mol=12.7,
        source="Engler et al. (1973), J. Am. Chem. Soc. 95, 5769",
        method="combustion calorimetry",
    ),
    ReferenceCompound(
        name="cyclodecane",
        smiles="C1CCCCCCCCC1",
        ring_sizes=[10],
        ring_count=1,
        strain_energy_kcal_mol=12.4,
        source="Engler et al. (1973), J. Am. Chem. Soc. 95, 5769",
        method="combustion calorimetry",
    ),

    # ---------- Polycyclic hydrocarbons ----------
    ReferenceCompound(
        name="norbornane",
        smiles="C1CC2CCC1C2",
        ring_sizes=[5, 5, 6],
        ring_count=3,
        strain_energy_kcal_mol=15.0,
        source="Steele et al. (1971), J. Chem. Thermodyn. 3, 843",
        method="combustion calorimetry",
        notes="Bridged bicyclic; two 5-membered + one 6-membered ring.",
    ),
    ReferenceCompound(
        name="adamantane",
        smiles="C1C2CC3CC1CC(C2)C3",
        ring_sizes=[6, 6, 6, 6],
        ring_count=4,
        strain_energy_kcal_mol=6.4,
        source="Clark et al. (1979), J. Chem. Thermodyn. 11, 457",
        method="combustion calorimetry",
        notes="Low-strain diamondoid cage.",
    ),
    ReferenceCompound(
        name="cubane",
        smiles="C12C3C4C1C5C2C3C45",
        ring_sizes=[4, 4, 4, 4, 4, 4],
        ring_count=6,
        strain_energy_kcal_mol=166.0,
        source="Kybet et al. (1965), J. Am. Chem. Soc. 87, 4865",
        method="combustion calorimetry",
        notes="Highly strained cage; all faces are cyclobutane-like.",
    ),
    ReferenceCompound(
        name="trans-decalin",
        smiles="C1CCC2CCCCC2C1",
        ring_sizes=[6, 6],
        ring_count=2,
        strain_energy_kcal_mol=3.1,
        source="Chang et al. (1971), J. Chem. Thermodyn. 3, 361",
        method="combustion calorimetry",
        notes="Fused 6/6 rings, chair-chair conformation.",
    ),
    ReferenceCompound(
        name="bicyclo[1.1.1]pentane",
        smiles="C1C2CC1C2",
        ring_sizes=[4, 4, 4],
        ring_count=3,
        strain_energy_kcal_mol=55.0,
        source="Wiberg & Fenoglio (1968), J. Am. Chem. Soc. 90, 3395",
        method="combustion calorimetry",
    ),

    # ---------- Heterocyclic and unsaturated ----------
    ReferenceCompound(
        name="tetrahydrofuran",
        smiles="C1CCOC1",
        ring_sizes=[5],
        ring_count=1,
        strain_energy_kcal_mol=6.2,
        source="estimated from cyclopentane analogue",
        method="estimated (cyclopentane analogue)",
        notes="Oxygen heterocycle; strain similar to cyclopentane.",
    ),
    ReferenceCompound(
        name="tetrahydropyran",
        smiles="C1CCOCC1",
        ring_sizes=[6],
        ring_count=1,
        strain_energy_kcal_mol=1.4,
        source="estimated from cyclohexane analogue",
        method="estimated (cyclohexane analogue)",
        notes="Oxygen heterocycle; low strain chair conformer.",
    ),
    ReferenceCompound(
        name="cyclopentene",
        smiles="C1CC=CC1",
        ring_sizes=[5],
        ring_count=1,
        strain_energy_kcal_mol=5.9,
        source="Allinger et al. (1990), J. Am. Chem. Soc. 112, 8293",
        method="computational (MM3)",
        notes="Slightly less strain than cyclopentane due to sp2 carbons.",
    ),

    # ---------- Small heterocycles ----------
    ReferenceCompound(
        name="aziridine",
        smiles="C1CN1",
        ring_sizes=[3],
        ring_count=1,
        strain_energy_kcal_mol=27.7,
        source="Dudev & Lim (1998), J. Am. Chem. Soc. 120, 4450",
        method="computational (high-level ab initio)",
        notes="Nitrogen analogue of cyclopropane; similar strain.",
    ),
    ReferenceCompound(
        name="azetidine",
        smiles="C1CNC1",
        ring_sizes=[4],
        ring_count=1,
        strain_energy_kcal_mol=25.7,
        source="Dudev & Lim (1998), J. Am. Chem. Soc. 120, 4450",
        method="computational (high-level ab initio)",
    ),
    ReferenceCompound(
        name="ethylene_oxide",
        smiles="C1CO1",
        ring_sizes=[3],
        ring_count=1,
        strain_energy_kcal_mol=27.3,
        source="Wiberg (1986), Angew. Chem. Int. Ed. 25, 312",
        method="combustion calorimetry",
    ),

    # ---------- Aromatics (zero strain by definition for calibration) ----------
    ReferenceCompound(
        name="benzene",
        smiles="c1ccccc1",
        ring_sizes=[6],
        ring_count=1,
        strain_energy_kcal_mol=0.0,
        source="aromatic; resonance-stabilized",
        method="definition",
        notes="Aromatic; no angle strain in planar hexagon.",
    ),
    ReferenceCompound(
        name="pyridine",
        smiles="c1ccncc1",
        ring_sizes=[6],
        ring_count=1,
        strain_energy_kcal_mol=0.0,
        source="aromatic heterocycle; resonance-stabilized",
        method="definition",
    ),
    ReferenceCompound(
        name="naphthalene",
        smiles="c1ccc2ccccc2c1",
        ring_sizes=[6, 6],
        ring_count=2,
        strain_energy_kcal_mol=0.0,
        source="fused aromatic; resonance-stabilized",
        method="definition",
    ),
]

# Build fast lookup maps
_SMILES_TO_REF: Dict[str, ReferenceCompound] = {}
_SIZE_TO_REFS: Dict[int, List[ReferenceCompound]] = {}


def _build_maps():
    """Populate lookup maps from _REFERENCE_COMPOUNDS."""
    for ref in _REFERENCE_COMPOUNDS:
        mol = Chem.MolFromSmiles(ref.smiles)
        if mol is None:
            continue
        can_smiles = Chem.MolToSmiles(mol, canonical=True)
        _SMILES_TO_REF[can_smiles] = ref
        for size in set(ref.ring_sizes):
            _SIZE_TO_REFS.setdefault(size, []).append(ref)


# Defer import to support module-level init
from rdkit import Chem


# ---------------------------------------------------------------------------
# Database class
# ---------------------------------------------------------------------------

class ReferenceDatabase:
    """Database of compounds with experimentally known ring strain energies.

    Provides lookup by SMILES, ring size, and name; exports calibration
    data for mapping MMFF94 results to the experimental scale.
    """

    def __init__(self):
        if not _SMILES_TO_REF:
            _build_maps()

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_by_smiles(self, smiles: str) -> Optional[ReferenceCompound]:
        """Look up a known compound by SMILES (after canonicalization)."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        can_smiles = Chem.MolToSmiles(mol, canonical=True)
        return _SMILES_TO_REF.get(can_smiles)

    def get_by_name(self, name: str) -> Optional[ReferenceCompound]:
        name_lower = name.strip().lower()
        for ref in _REFERENCE_COMPOUNDS:
            if ref.name.lower() == name_lower:
                return ref
        return None

    def get_by_ring_size(self, ring_size: int) -> List[ReferenceCompound]:
        return _SIZE_TO_REFS.get(ring_size, [])

    def get_all(self) -> List[ReferenceCompound]:
        return list(_REFERENCE_COMPOUNDS)

    def get_monocyclics(self) -> List[ReferenceCompound]:
        return [r for r in _REFERENCE_COMPOUNDS if r.ring_count == 1]

    def get_cycloalkanes(self) -> List[ReferenceCompound]:
        """Return only simple carbocyclic monocyclic references."""
        cycloalkanes = []
        for r in _REFERENCE_COMPOUNDS:
            if r.ring_count != 1 or r.ring_sizes[0] > 8:
                continue
            mol = Chem.MolFromSmiles(r.smiles)
            if mol is None:
                continue
            # All atoms must be carbon
            if not all(a.GetAtomicNum() == 6 for a in mol.GetAtoms()):
                continue
            # Exclude aromatics (lowercase 'c' in SMILES means aromatic carbon)
            if any(a.GetIsAromatic() for a in mol.GetAtoms()):
                continue
            cycloalkanes.append(r)
        return cycloalkanes

    # ------------------------------------------------------------------
    # Calibration data export
    # ------------------------------------------------------------------

    def get_calibration_pairs(
        self,
    ) -> List[Tuple[ReferenceCompound, str]]:
        """Return (ref_compound, canonical_smiles) pairs suitable for calibration.

        Filters to monocyclic carbocycles (3-8 membered) which form the
        core calibration set.
        """
        pairs = []
        for ref in self.get_cycloalkanes():
            mol = Chem.MolFromSmiles(ref.smiles)
            if mol is None:
                continue
            can_smiles = Chem.MolToSmiles(mol, canonical=True)
            pairs.append((ref, can_smiles))
        return pairs

    def export_calibration_curve(
        self,
    ) -> Dict[int, List[Tuple[str, float]]]:
        """Export calibration data organized by ring size.

        Returns:
            {ring_size: [(smiles, exp_strain_kcal_mol), ...]}
        """
        curve: Dict[int, List[Tuple[str, float]]] = {}
        for ref in _REFERENCE_COMPOUNDS:
            for size in set(ref.ring_sizes):
                curve.setdefault(size, []).append(
                    (ref.smiles, ref.strain_energy_kcal_mol)
                )
        return curve

    def get_experimental_map(self) -> Dict[str, float]:
        """Return dict mapping canonical SMILES -> experimental strain (kcal/mol)."""
        exp_map = {}
        for ref in _REFERENCE_COMPOUNDS:
            mol = Chem.MolFromSmiles(ref.smiles)
            if mol is None:
                continue
            can = Chem.MolToSmiles(mol, canonical=True)
            exp_map[can] = ref.strain_energy_kcal_mol
        return exp_map

    # ------------------------------------------------------------------
    # Add custom data
    # ------------------------------------------------------------------

    def add_compound(self, compound: ReferenceCompound) -> None:
        """Add a custom reference compound to the database."""
        _REFERENCE_COMPOUNDS.append(compound)
        mol = Chem.MolFromSmiles(compound.smiles)
        if mol is not None:
            can = Chem.MolToSmiles(mol, canonical=True)
            _SMILES_TO_REF[can] = compound
            for size in set(compound.ring_sizes):
                _SIZE_TO_REFS.setdefault(size, []).append(compound)
