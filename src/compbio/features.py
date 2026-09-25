"""Sequence features for genomic windows and protein sequences."""

from __future__ import annotations

import numpy as np

from compbio.constants import (
    ACIDIC_RESIDUES,
    AMINO_ACIDS,
    BASIC_RESIDUES,
    CHOU_FASMAN,
    GENETIC_CODE,
    KYTE_DOOLITTLE,
    PKA_C_TERMINUS,
    PKA_N_TERMINUS,
    PKA_SIDE,
    RESIDUE_MASS,
    SENSE_CODONS,
    STOP_CODONS,
    SYNONYMOUS_CODONS,
    WATER_MASS,
)

GENOMIC_STRUCTURE_FEATURES = frozenset(
    {
        "stop_density_f0",
        "stop_density_f1",
        "stop_density_f2",
        "longest_orf_fraction",
    }
)

PROTEOMIC_COMPOSITION_PREFIX = "aa_"


def clean_dna(sequence: str) -> str:
    cleaned = "".join(sequence.split()).upper().replace("U", "T")
    invalid = set(cleaned) - set("ACGT")
    if invalid:
        raise ValueError(f"Invalid DNA symbols: {sorted(invalid)}")
    if not cleaned:
        raise ValueError("DNA sequence is empty")
    return cleaned


def clean_protein(sequence: str) -> str:
    cleaned = "".join(sequence.split()).upper()
    invalid = set(cleaned) - set(AMINO_ACIDS)
    if invalid:
        raise ValueError(f"Invalid amino-acid symbols: {sorted(invalid)}")
    if not cleaned:
        raise ValueError("Protein sequence is empty")
    return cleaned


def translate(sequence: str) -> str:
    """Translate frame 0 until the first in-frame stop codon."""
    dna = clean_dna(sequence)
    protein: list[str] = []
    for index in range(0, len(dna) - 2, 3):
        amino_acid = GENETIC_CODE[dna[index : index + 3]]
        if amino_acid == "*":
            break
        protein.append(amino_acid)
    return "".join(protein)


def gc_content(sequence: str) -> float:
    dna = clean_dna(sequence)
    return sum(base in "GC" for base in dna) / len(dna)


def _skew(count_a: int, count_b: int) -> float:
    total = count_a + count_b
    if total == 0:
        return 0.0
    return (count_a - count_b) / total


def positional_gc(sequence: str) -> tuple[float, float, float]:
    dna = clean_dna(sequence)
    columns = [[] for _ in range(3)]
    for index in range(0, len(dna) - 2, 3):
        codon = dna[index : index + 3]
        for position in range(3):
            columns[position].append(codon[position])
    values = []
    for column in columns:
        if not column:
            values.append(0.0)
        else:
            values.append(sum(base in "GC" for base in column) / len(column))
    return values[0], values[1], values[2]


def _codons_in_frame(sequence: str, frame: int) -> list[str]:
    return [
        sequence[index : index + 3]
        for index in range(frame, len(sequence) - 2, 3)
    ]


def stop_density(sequence: str, frame: int) -> float:
    dna = clean_dna(sequence)
    codons = _codons_in_frame(dna, frame)
    if not codons:
        return 0.0
    return sum(codon in STOP_CODONS for codon in codons) / len(codons)


def longest_orf_fraction(sequence: str) -> float:
    """Longest stop-free stretch in any frame, as a fraction of sequence length."""
    dna = clean_dna(sequence)
    best = 0
    for frame in range(3):
        current = 0
        index = frame
        while index + 3 <= len(dna):
            if dna[index : index + 3] in STOP_CODONS:
                best = max(best, current)
                current = 0
            else:
                current += 3
            index += 3
        best = max(best, current)
    return best / len(dna)


def cpg_observed_expected(sequence: str) -> float:
    dna = clean_dna(sequence)
    cytosine = dna.count("C")
    guanine = dna.count("G")
    if cytosine == 0 or guanine == 0:
        return 0.0
    cg_pairs = sum(1 for index in range(len(dna) - 1) if dna[index : index + 2] == "CG")
    return (cg_pairs * len(dna)) / (cytosine * guanine)


def _sense_codon_counts(sequence: str) -> np.ndarray:
    dna = clean_dna(sequence)
    index = {codon: position for position, codon in enumerate(SENSE_CODONS)}
    counts = np.zeros(len(SENSE_CODONS), dtype=float)
    for codon in _codons_in_frame(dna, 0):
        if codon in index:
            counts[index[codon]] += 1
    return counts


def codon_entropy(sequence: str) -> float:
    """Shannon entropy, in bits, of the sense-codon distribution in frame 0."""
    counts = _sense_codon_counts(sequence)
    total = counts.sum()
    if total == 0:
        return 0.0
    probabilities = counts[counts > 0] / total
    return float(-(probabilities * np.log2(probabilities)).sum())


def effective_number_of_codons(sequence: str) -> float:
    """Wright's Nc. Range is clipped to the theoretical 20-61 interval.

    Amino acids with fewer than two observations, or a non-positive
    homozygosity estimate, are left out of their synonymous-family average.
    Missing families fall back to the mean homozygosity of the families
    that could be estimated.
    """
    dna = clean_dna(sequence)
    observed: dict[str, int] = {}
    for codon in _codons_in_frame(dna, 0):
        if codon in STOP_CODONS:
            continue
        observed[codon] = observed.get(codon, 0) + 1

    family_values: dict[int, list[float]] = {2: [], 3: [], 4: [], 6: []}
    for amino_acid, codons in SYNONYMOUS_CODONS.items():
        family_size = len(codons)
        if family_size == 1:
            continue
        count = sum(observed.get(codon, 0) for codon in codons)
        if count < 2:
            continue
        probabilities = np.array(
            [observed.get(codon, 0) / count for codon in codons], dtype=float
        )
        homozygosity = (count * float(np.square(probabilities).sum()) - 1.0) / (count - 1.0)
        if homozygosity <= 0.01:
            continue
        family_values[family_size].append(homozygosity)

    available = [value for values in family_values.values() for value in values]
    if not available:
        return 61.0
    fallback = float(np.mean(available))

    def family_average(size: int) -> float:
        values = family_values[size]
        return float(np.mean(values)) if values else fallback

    nc = (
        2.0
        + 9.0 / family_average(2)
        + 1.0 / family_average(3)
        + 5.0 / family_average(4)
        + 3.0 / family_average(6)
    )
    return float(min(61.0, max(20.0, nc)))


def genomic_feature_names() -> list[str]:
    return [
        "gc",
        "at_skew",
        "gc_skew",
        "gc1",
        "gc2",
        "gc3",
        "stop_density_f0",
        "stop_density_f1",
        "stop_density_f2",
        "longest_orf_fraction",
        "codon_entropy",
        "effective_number_of_codons",
        "cpg_oe",
        *[f"codon_{codon}" for codon in SENSE_CODONS],
    ]


def genomic_feature_vector(sequence: str) -> np.ndarray:
    dna = clean_dna(sequence)
    gc1, gc2, gc3 = positional_gc(dna)
    counts = _sense_codon_counts(dna)
    total = counts.sum()
    frequencies = counts / total if total else counts
    scalars = [
        gc_content(dna),
        _skew(dna.count("A"), dna.count("T")),
        _skew(dna.count("G"), dna.count("C")),
        gc1,
        gc2,
        gc3,
        stop_density(dna, 0),
        stop_density(dna, 1),
        stop_density(dna, 2),
        longest_orf_fraction(dna),
        codon_entropy(dna),
        effective_number_of_codons(dna),
        cpg_observed_expected(dna),
    ]
    return np.concatenate([np.asarray(scalars, dtype=float), frequencies])


def gravy(sequence: str) -> float:
    protein = clean_protein(sequence)
    return sum(KYTE_DOOLITTLE[residue] for residue in protein) / len(protein)


def aromaticity(sequence: str) -> float:
    protein = clean_protein(sequence)
    return sum(residue in "FYW" for residue in protein) / len(protein)


def peptide_mass(sequence: str) -> float:
    protein = clean_protein(sequence)
    return sum(RESIDUE_MASS[residue] for residue in protein) + WATER_MASS


def mean_residue_mass(sequence: str) -> float:
    protein = clean_protein(sequence)
    return sum(RESIDUE_MASS[residue] for residue in protein) / len(protein)


def net_charge(sequence: str, ph: float) -> float:
    protein = clean_protein(sequence)
    charge = 1.0 / (1.0 + 10 ** (ph - PKA_N_TERMINUS))
    charge -= 1.0 / (1.0 + 10 ** (PKA_C_TERMINUS - ph))
    for residue in protein:
        if residue in BASIC_RESIDUES:
            charge += 1.0 / (1.0 + 10 ** (ph - PKA_SIDE[residue]))
        elif residue in ACIDIC_RESIDUES:
            charge -= 1.0 / (1.0 + 10 ** (PKA_SIDE[residue] - ph))
    return charge


def isoelectric_point(sequence: str) -> float:
    """pH at which the estimated peptide net charge is zero."""
    low = 0.0
    high = 14.0
    for _ in range(60):
        midpoint = (low + high) / 2.0
        if net_charge(sequence, midpoint) > 0:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2.0


def _mean_propensity(sequence: str, kind: int) -> float:
    protein = clean_protein(sequence)
    return sum(CHOU_FASMAN[residue][kind] for residue in protein) / len(protein)


def composition_entropy(sequence: str) -> float:
    protein = clean_protein(sequence)
    counts = np.array([protein.count(residue) for residue in AMINO_ACIDS], dtype=float)
    probabilities = counts[counts > 0] / len(protein)
    return float(-(probabilities * np.log2(probabilities)).sum())


def proteomic_feature_names() -> list[str]:
    return [
        *[f"aa_{residue}" for residue in AMINO_ACIDS],
        "gravy",
        "aromaticity",
        "mean_residue_mass",
        "isoelectric_point",
        "charge_ph7",
        "helix_propensity",
        "sheet_propensity",
        "turn_propensity",
        "composition_entropy",
    ]


def proteomic_feature_vector(sequence: str) -> np.ndarray:
    protein = clean_protein(sequence)
    composition = np.array(
        [protein.count(residue) / len(protein) for residue in AMINO_ACIDS],
        dtype=float,
    )
    properties = np.array(
        [
            gravy(protein),
            aromaticity(protein),
            mean_residue_mass(protein),
            isoelectric_point(protein),
            net_charge(protein, 7.0),
            _mean_propensity(protein, 0),
            _mean_propensity(protein, 1),
            _mean_propensity(protein, 2),
            composition_entropy(protein),
        ],
        dtype=float,
    )
    return np.concatenate([composition, properties])


def is_composition_feature(name: str) -> bool:
    return name.startswith(PROTEOMIC_COMPOSITION_PREFIX) or name == "composition_entropy"
