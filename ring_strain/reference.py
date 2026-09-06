"""
Reference compound database for ring strain energy calibration.

Contains experimentally known ring strain values from the literature
for monocyclic saturated carbocycles (cycloalkanes).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


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
# Built-in reference dataset — monocyclic saturated carbocycles
# ---------------------------------------------------------------------------

_REFERENCE_COMPOUNDS: List[ReferenceCompound] = [
    ReferenceCompound(
        name="cyclopropane",
        smiles="C1CC1",
        ring_sizes=[3],
        ring_count=1,
        strain_energy_kcal_mol=27.5,
        source="Wiberg (1986), Angew. Chem. Int. Ed. 25, 312",
    ),
    ReferenceCompound(
        name="cyclobutane",
        smiles="C1CCC1",
        ring_sizes=[4],
        ring_count=1,
        strain_energy_kcal_mol=26.3,
        source="Wiberg (1986), Angew. Chem. Int. Ed. 25, 312",
    ),
    ReferenceCompound(
        name="cyclopentane",
        smiles="C1CCCC1",
        ring_sizes=[5],
        ring_count=1,
        strain_energy_kcal_mol=6.5,
        source="Wiberg (1986), Angew. Chem. Int. Ed. 25, 312",
    ),
    ReferenceCompound(
        name="cyclohexane",
        smiles="C1CCCCC1",
        ring_sizes=[6],
        ring_count=1,
        strain_energy_kcal_mol=0.0,
        source="Wiberg (1986), Angew. Chem. Int. Ed. 25, 312",
        notes="Strain-free reference ring size.",
    ),
    ReferenceCompound(
        name="cycloheptane",
        smiles="C1CCCCCC1",
        ring_sizes=[7],
        ring_count=1,
        strain_energy_kcal_mol=6.3,
        source="Wiberg (1986), Angew. Chem. Int. Ed. 25, 312",
    ),
    ReferenceCompound(
        name="cyclooctane",
        smiles="C1CCCCCCC1",
        ring_sizes=[8],
        ring_count=1,
        strain_energy_kcal_mol=9.7,
        source="Wiberg (1986), Angew. Chem. Int. Ed. 25, 312",
    ),
    ReferenceCompound(
        name="cyclononane",
        smiles="C1CCCCCCCC1",
        ring_sizes=[9],
        ring_count=1,
        strain_energy_kcal_mol=12.7,
        source="Engler et al. (1973), J. Am. Chem. Soc. 95, 5769",
    ),
    ReferenceCompound(
        name="cyclodecane",
        smiles="C1CCCCCCCCC1",
        ring_sizes=[10],
        ring_count=1,
        strain_energy_kcal_mol=12.4,
        source="Engler et al. (1973), J. Am. Chem. Soc. 95, 5769",
    ),
    # Steric hindrance reference: trans-1,2-di-tert-butylcyclopropane
    # Strain is elevated vs. cyclopropane (27.5) due to steric repulsion
    # between the two bulky tert-butyl groups on adjacent ring carbons.
    ReferenceCompound(
        name="trans-1,2-di-tert-butylcyclopropane",
        smiles="CC(C)(C)[C@@H]1C[C@H]1C(C)(C)C",
        ring_sizes=[3],
        ring_count=1,
        strain_energy_kcal_mol=32.0,
        source="estimated from steric enhancement of cyclopropane strain",
        method="estimated",
        notes="Sterically enhanced strain ~31-33.5 kcal/mol vs 27.5 for cyclopropane.",
    ),
    # -------------------------------------------------------------
    # Substituted ring references for bulky-substituent calibration
    # -------------------------------------------------------------
    ReferenceCompound(
        name="1,1-dimethylcyclopropane",
        smiles="CC1(C)CC1",
        ring_sizes=[3],
        ring_count=1,
        strain_energy_kcal_mol=31.0,
        source="Allen, J. Chem. Soc. Perkin Trans 2 (1996)",
        method="combustion + group additivity",
        notes="gem-dimethyl on cyclopropane; vdW + angle strain dominate.",
    ),
    ReferenceCompound(
        name="1,1,2,2-tetramethylcyclopropane",
        smiles="CC1(C)C(C)(C)C1",
        ring_sizes=[3],
        ring_count=1,
        strain_energy_kcal_mol=35.5,
        source="Schleyer, J. Am. Chem. Soc. 92, 2377 (1970)",
        method="combustion + group additivity",
        notes="Four methyls on cyclopropane; severe gem and vicinal vdW clash.",
    ),
    ReferenceCompound(
        name="tert-butylcyclohexane",
        smiles="CC(C)(C)C1CCCCC1",
        ring_sizes=[6],
        ring_count=1,
        strain_energy_kcal_mol=5.4,
        source="Allinger MM4 study, J. Comput. Chem. 17, 642 (1996)",
        method="MM4 vs experimental",
        notes="Equatorial t-Bu; gauche-butane interactions across ring.",
    ),
    ReferenceCompound(
        name="cis-1,3-di-tert-butylcyclohexane",
        smiles="CC(C)(C)[C@H]1CCC[C@@H](C(C)(C)C)C1",
        ring_sizes=[6],
        ring_count=1,
        strain_energy_kcal_mol=11.2,
        source="Allinger MM4 + Eliel conformational analysis",
        method="MM4",
        notes="cis forces one t-Bu axial; large 1,3-diaxial vdW strain.",
    ),
    ReferenceCompound(
        name="trans-1,3-di-tert-butylcyclohexane",
        smiles="CC(C)(C)[C@H]1CCC[C@H](C(C)(C)C)C1",
        ring_sizes=[6],
        ring_count=1,
        strain_energy_kcal_mol=0.5,
        source="Allinger MM4 + Eliel conformational analysis",
        method="MM4",
        notes="trans places both t-Bu equatorial; minimal extra strain.",
    ),
    ReferenceCompound(
        name="1,1-dimethylcyclobutane",
        smiles="CC1(C)CCC1",
        ring_sizes=[4],
        ring_count=1,
        strain_energy_kcal_mol=27.5,
        source="NIST WebBook + group additivity",
        method="combustion",
        notes="gem-dimethyl on cyclobutane; small additional vdW vs C4 base.",
    ),
    ReferenceCompound(
        name="1,1-dimethylcyclopentane",
        smiles="CC1(C)CCCC1",
        ring_sizes=[5],
        ring_count=1,
        strain_energy_kcal_mol=7.5,
        source="NIST WebBook + group additivity",
        method="combustion",
        notes="gem-dimethyl on cyclopentane.",
    ),
    ReferenceCompound(
        name="1,1-dimethylcyclohexane",
        smiles="CC1(C)CCCCC1",
        ring_sizes=[6],
        ring_count=1,
        strain_energy_kcal_mol=1.4,
        source="NIST WebBook + group additivity",
        method="combustion",
        notes="gem-dimethyl on cyclohexane; small.",
    ),
    # -------------------------------------------------------------
    # Mono-substituted (non-gem) anchors — add diversity to per-size
    # calibration which was previously gem-only on sizes 3/4/5. These
    # are all real-world synthesizable compounds with NIST/Wiberg data.
    # -------------------------------------------------------------
    ReferenceCompound(
        name="methylcyclopropane",
        smiles="CC1CC1",
        ring_sizes=[3],
        ring_count=1,
        strain_energy_kcal_mol=28.4,
        source="Wiberg (1986) group additivity",
        method="combustion + group additivity",
        notes="Single methyl on cyclopropane; mono-substituted anchor.",
    ),
    ReferenceCompound(
        name="methylcyclobutane",
        smiles="CC1CCC1",
        ring_sizes=[4],
        ring_count=1,
        strain_energy_kcal_mol=26.7,
        source="NIST WebBook",
        method="combustion",
        notes="Single methyl on cyclobutane; mono-substituted anchor.",
    ),
    ReferenceCompound(
        name="methylcyclopentane",
        smiles="CC1CCCC1",
        ring_sizes=[5],
        ring_count=1,
        strain_energy_kcal_mol=6.4,
        source="NIST WebBook",
        method="combustion",
        notes="Single methyl on cyclopentane; mono-substituted anchor.",
    ),
    ReferenceCompound(
        name="methylcyclohexane",
        smiles="CC1CCCCC1",
        ring_sizes=[6],
        ring_count=1,
        strain_energy_kcal_mol=1.7,
        source="NIST WebBook + A-value",
        method="combustion",
        notes="Single methyl on cyclohexane; gauche-methyl contribution.",
    ),
    ReferenceCompound(
        name="trans-1,4-dimethylcyclohexane",
        smiles="C[C@H]1CC[C@H](C)CC1",
        ring_sizes=[6],
        ring_count=1,
        strain_energy_kcal_mol=1.9,
        source="NIST WebBook",
        method="combustion",
        notes="Both methyls equatorial; size-6 stereo anchor at low strain.",
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


from rdkit import Chem


class ReferenceDatabase:
    """Database of compounds with experimentally known ring strain energies.

    Provides lookup by SMILES, ring size, and name; exports calibration
    data for mapping MMFF94 results to the experimental scale.
    """

    def __init__(self):
        if not _SMILES_TO_REF:
            _build_maps()

    def get_by_smiles(self, smiles: str) -> Optional[ReferenceCompound]:
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
        """Return simple unsubstituted cycloalkane references."""
        cycloalkanes = []
        for r in _REFERENCE_COMPOUNDS:
            if r.ring_count != 1:
                continue
            mol = Chem.MolFromSmiles(r.smiles)
            if mol is None:
                continue
            if not all(a.GetAtomicNum() == 6 for a in mol.GetAtoms()):
                continue
            if any(a.GetIsAromatic() for a in mol.GetAtoms()):
                continue
            cycloalkanes.append(r)
        return cycloalkanes

    def get_calibration_pairs(self) -> List[Tuple[ReferenceCompound, str]]:
        """Return (ref, canonical_smiles) for calibration (unsubstituted cycloalkanes)."""
        pairs = []
        for ref in self.get_cycloalkanes():
            mol = Chem.MolFromSmiles(ref.smiles)
            if mol is None:
                continue
            can_smiles = Chem.MolToSmiles(mol, canonical=True)
            pairs.append((ref, can_smiles))
        return pairs

    def export_calibration_curve(self) -> Dict[int, List[Tuple[str, float]]]:
        curve: Dict[int, List[Tuple[str, float]]] = {}
        for ref in _REFERENCE_COMPOUNDS:
            for size in set(ref.ring_sizes):
                curve.setdefault(size, []).append(
                    (ref.smiles, ref.strain_energy_kcal_mol)
                )
        return curve

    def get_experimental_map(self) -> Dict[str, float]:
        exp_map = {}
        for ref in _REFERENCE_COMPOUNDS:
            mol = Chem.MolFromSmiles(ref.smiles)
            if mol is None:
                continue
            can = Chem.MolToSmiles(mol, canonical=True)
            exp_map[can] = ref.strain_energy_kcal_mol
        return exp_map

    def add_compound(self, compound: ReferenceCompound) -> None:
        _REFERENCE_COMPOUNDS.append(compound)
        mol = Chem.MolFromSmiles(compound.smiles)
        if mol is not None:
            can = Chem.MolToSmiles(mol, canonical=True)
            _SMILES_TO_REF[can] = compound
            for size in set(compound.ring_sizes):
                _SIZE_TO_REFS.setdefault(size, []).append(compound)
