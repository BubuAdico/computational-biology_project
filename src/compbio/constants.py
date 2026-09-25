"""Reference tables for sequence statistics.

Codon weights are rounded per-thousand frequencies for human coding sequence,
in the form published by the Kazusa codon usage compilation. Propensity,
hydrophobicity, pKa, and residue-mass tables are the standard textbook scales
used to compute sequence features. They are measurement constants, not a
recipe for producing biological material.
"""

from __future__ import annotations

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

# DNA codon -> amino acid. "*" is a stop codon.
GENETIC_CODE: dict[str, str] = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

STOP_CODONS = frozenset({"TAA", "TAG", "TGA"})

# Counts per thousand codons, human CDS (Kazusa-style table, rounded).
HUMAN_CODON_PER_THOUSAND: dict[str, float] = {
    "TTT": 17.6, "TTC": 20.3, "TTA": 7.7, "TTG": 12.9,
    "TCT": 15.2, "TCC": 17.7, "TCA": 12.2, "TCG": 4.4,
    "TAT": 12.2, "TAC": 15.3, "TAA": 1.0, "TAG": 0.8,
    "TGT": 10.6, "TGC": 12.6, "TGA": 1.6, "TGG": 13.2,
    "CTT": 13.2, "CTC": 19.6, "CTA": 7.2, "CTG": 39.6,
    "CCT": 17.5, "CCC": 19.8, "CCA": 16.9, "CCG": 6.9,
    "CAT": 10.9, "CAC": 15.1, "CAA": 12.3, "CAG": 34.2,
    "CGT": 4.5, "CGC": 10.4, "CGA": 6.2, "CGG": 11.4,
    "ATT": 16.0, "ATC": 20.8, "ATA": 7.5, "ATG": 22.0,
    "ACT": 13.1, "ACC": 18.9, "ACA": 15.1, "ACG": 6.1,
    "AAT": 17.0, "AAC": 19.1, "AAA": 24.4, "AAG": 31.9,
    "AGT": 12.1, "AGC": 19.5, "AGA": 12.2, "AGG": 12.0,
    "GTT": 11.0, "GTC": 14.5, "GTA": 7.1, "GTG": 28.1,
    "GCT": 18.4, "GCC": 27.7, "GCA": 15.8, "GCG": 7.4,
    "GAT": 21.8, "GAC": 25.1, "GAA": 29.0, "GAG": 39.6,
    "GGT": 10.8, "GGC": 22.2, "GGA": 16.5, "GGG": 16.5,
}

SENSE_CODONS: tuple[str, ...] = tuple(
    codon for codon in sorted(GENETIC_CODE) if GENETIC_CODE[codon] != "*"
)

SYNONYMOUS_CODONS: dict[str, tuple[str, ...]] = {
    aa: tuple(codon for codon in sorted(GENETIC_CODE) if GENETIC_CODE[codon] == aa)
    for aa in AMINO_ACIDS
}

# Kyte and Doolittle, 1982. Positive values are hydrophobic.
KYTE_DOOLITTLE: dict[str, float] = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
    "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}

# Chou and Fasman, 1978. Conformational parameters (helix, sheet, turn).
CHOU_FASMAN: dict[str, tuple[float, float, float]] = {
    "A": (1.42, 0.83, 0.66), "C": (0.70, 1.19, 1.19), "D": (1.01, 0.54, 1.46),
    "E": (1.51, 0.37, 0.74), "F": (1.13, 1.38, 0.60), "G": (0.57, 0.75, 1.56),
    "H": (1.00, 0.87, 0.95), "I": (1.08, 1.60, 0.47), "K": (1.16, 0.74, 1.01),
    "L": (1.21, 1.30, 0.59), "M": (1.45, 1.05, 0.60), "N": (0.67, 0.89, 1.56),
    "P": (0.57, 0.55, 1.52), "Q": (1.11, 1.10, 0.98), "R": (0.98, 0.93, 0.95),
    "S": (0.77, 0.75, 1.43), "T": (0.83, 1.19, 0.96), "V": (1.06, 1.70, 0.50),
    "W": (1.08, 1.37, 0.96), "Y": (0.69, 1.47, 1.14),
}

# Textbook side-chain and terminal pKa values for computational pI.
PKA_SIDE: dict[str, float] = {
    "C": 8.18, "D": 3.65, "E": 4.25, "H": 6.00,
    "K": 10.53, "R": 12.48, "Y": 10.07,
}
PKA_N_TERMINUS = 9.69
PKA_C_TERMINUS = 2.34
BASIC_RESIDUES = frozenset({"H", "K", "R"})
ACIDIC_RESIDUES = frozenset({"C", "D", "E", "Y"})

# Average isotopic residue masses (Da), already corrected by one water.
RESIDUE_MASS: dict[str, float] = {
    "A": 71.0788, "C": 103.1388, "D": 115.0886, "E": 129.1155, "F": 147.1766,
    "G": 57.0519, "H": 137.1411, "I": 113.1594, "K": 128.1741, "L": 113.1594,
    "M": 131.1926, "N": 114.1038, "P": 97.1167, "Q": 128.1307, "R": 156.1875,
    "S": 87.0782, "T": 101.1051, "V": 99.1326, "W": 186.2132, "Y": 163.1760,
}
WATER_MASS = 18.0153
