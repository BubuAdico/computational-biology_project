"""Simulated genomic windows and protein sequences with known labels.

Coding DNA is drawn from a human codon-usage model. Noncoding DNA is drawn
from a base-composition model with the same expected GC, so overall GC is not
a shortcut. Protein classes differ by amino-acid frequency profiles that
follow well-known biophysical tendencies: hydrophobic residues in
membrane-like sequences, and disorder-associated residues in the disordered
class. Every sequence is synthetic.
"""

from __future__ import annotations

import numpy as np

from compbio.constants import AMINO_ACIDS, GENETIC_CODE, HUMAN_CODON_PER_THOUSAND, KYTE_DOOLITTLE, SENSE_CODONS
from compbio.io import Record

FRAMESHIFT_RATE = 0.30
EMBEDDED_ORF_RATE = 0.25
DIRICHLET_CONCENTRATION = 45.0

# Multipliers applied to the human amino-acid background, then renormalized.
GLOBULAR_FACTORS: dict[str, float] = {
    "A": 0.95, "C": 1.10, "D": 1.30, "E": 1.30, "F": 0.95,
    "G": 0.90, "H": 1.25, "I": 0.85, "K": 1.25, "L": 0.80,
    "M": 1.00, "N": 1.15, "P": 0.90, "Q": 1.10, "R": 1.15,
    "S": 1.05, "T": 1.00, "V": 0.90, "W": 0.95, "Y": 1.05,
}
MEMBRANE_FACTORS: dict[str, float] = {
    "A": 1.50, "C": 1.10, "D": 0.22, "E": 0.22, "F": 2.10,
    "G": 1.40, "H": 0.55, "I": 2.30, "K": 0.28, "L": 2.60,
    "M": 1.70, "N": 0.40, "P": 0.45, "Q": 0.40, "R": 0.30,
    "S": 0.65, "T": 0.80, "V": 2.20, "W": 1.35, "Y": 1.10,
}
DISORDERED_FACTORS: dict[str, float] = {
    "A": 1.45, "C": 0.25, "D": 1.15, "E": 2.05, "F": 0.35,
    "G": 1.30, "H": 1.00, "I": 0.35, "K": 1.85, "L": 0.45,
    "M": 0.50, "N": 0.90, "P": 2.20, "Q": 1.85, "R": 1.25,
    "S": 1.85, "T": 1.15, "V": 0.40, "W": 0.20, "Y": 0.40,
}

PROTEIN_CLASSES = ("globular", "membrane", "disordered")


def sense_codon_probabilities() -> tuple[np.ndarray, np.ndarray]:
    codons = np.array(SENSE_CODONS)
    weights = np.array([HUMAN_CODON_PER_THOUSAND[codon] for codon in SENSE_CODONS], dtype=float)
    weights /= weights.sum()
    return codons, weights


def expected_coding_gc() -> float:
    codons, weights = sense_codon_probabilities()
    gc_fraction = np.array(
        [(codon.count("G") + codon.count("C")) / 3.0 for codon in codons],
        dtype=float,
    )
    return float(np.dot(weights, gc_fraction))


def human_aa_frequency() -> dict[str, float]:
    totals = {residue: 0.0 for residue in AMINO_ACIDS}
    for codon, weight in HUMAN_CODON_PER_THOUSAND.items():
        amino_acid = GENETIC_CODE[codon]
        if amino_acid == "*":
            continue
        totals[amino_acid] += weight
    scale = sum(totals.values())
    return {residue: totals[residue] / scale for residue in AMINO_ACIDS}


def reweight(base: dict[str, float], factors: dict[str, float]) -> dict[str, float]:
    raw = {residue: base[residue] * factors[residue] for residue in AMINO_ACIDS}
    scale = sum(raw.values())
    return {residue: raw[residue] / scale for residue in AMINO_ACIDS}


def expected_gravy(frequency: dict[str, float]) -> float:
    return sum(frequency[residue] * KYTE_DOOLITTLE[residue] for residue in AMINO_ACIDS)


def protein_class_profiles() -> dict[str, dict[str, float]]:
    background = human_aa_frequency()
    return {
        "globular": reweight(background, GLOBULAR_FACTORS),
        "membrane": reweight(background, MEMBRANE_FACTORS),
        "disordered": reweight(background, DISORDERED_FACTORS),
    }


def _sample_coding_codons(rng: np.random.Generator, n_codons: int) -> str:
    codons, weights = sense_codon_probabilities()
    chosen = rng.choice(codons, size=n_codons, replace=True, p=weights)
    return "".join(str(codon) for codon in chosen)


def simulate_coding(
    rng: np.random.Generator,
    n_codons: int,
    frameshift_rate: float = FRAMESHIFT_RATE,
) -> str:
    sequence = _sample_coding_codons(rng, n_codons)
    if frameshift_rate > 0 and rng.random() < frameshift_rate and len(sequence) > 6:
        position = int(rng.integers(3, len(sequence) - 1))
        sequence = sequence[:position] + sequence[position + 1 :]
    return sequence


def simulate_noncoding(
    rng: np.random.Generator,
    length: int,
    target_gc: float | None = None,
    embedded_orf_rate: float = EMBEDDED_ORF_RATE,
) -> str:
    if length < 3:
        raise ValueError("Noncoding sequences must be at least 3 bases")
    gc = expected_coding_gc() if target_gc is None else target_gc
    at_each = (1.0 - gc) / 2.0
    gc_each = gc / 2.0
    bases = np.array(list("ACGT"))
    probabilities = np.array([at_each, gc_each, gc_each, at_each])
    sequence = "".join(rng.choice(bases, size=length, replace=True, p=probabilities))
    if embedded_orf_rate > 0 and rng.random() < embedded_orf_rate:
        n_codons = int(rng.integers(12, 28))
        orf = _sample_coding_codons(rng, n_codons)
        if len(orf) < length:
            start = int(rng.integers(0, length - len(orf) + 1))
            sequence = sequence[:start] + orf + sequence[start + len(orf) :]
    return sequence


def simulate_protein(
    rng: np.random.Generator,
    frequency: dict[str, float],
    length: int,
    concentration: float = DIRICHLET_CONCENTRATION,
) -> str:
    alpha = np.array([frequency[residue] for residue in AMINO_ACIDS], dtype=float)
    alpha = np.maximum(alpha * concentration, 1e-3)
    probabilities = rng.dirichlet(alpha)
    residues = rng.choice(np.array(list(AMINO_ACIDS)), size=length, replace=True, p=probabilities)
    return "".join(residues.tolist())


def make_genomic_dataset(n_per_class: int, rng: np.random.Generator) -> list[Record]:
    if n_per_class < 1:
        raise ValueError("n_per_class must be positive")
    records: list[Record] = []
    for index in range(n_per_class):
        n_codons = int(rng.integers(40, 101))
        records.append((f"coding_{index:04d}", "coding", simulate_coding(rng, n_codons)))
        records.append(
            (f"noncoding_{index:04d}", "noncoding", simulate_noncoding(rng, n_codons * 3))
        )
    order = rng.permutation(len(records))
    return [records[int(index)] for index in order]


def make_proteomic_dataset(n_per_class: int, rng: np.random.Generator) -> list[Record]:
    if n_per_class < 1:
        raise ValueError("n_per_class must be positive")
    profiles = protein_class_profiles()
    records: list[Record] = []
    for label in PROTEIN_CLASSES:
        for index in range(n_per_class):
            length = int(rng.integers(150, 351))
            sequence = simulate_protein(rng, profiles[label], length)
            records.append((f"{label}_{index:04d}", label, sequence))
    order = rng.permutation(len(records))
    return [records[int(index)] for index in order]
