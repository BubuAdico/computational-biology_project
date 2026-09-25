"""Tests for sequence statistics, simulation, and a small model run."""

from __future__ import annotations

import numpy as np

from compbio.constants import AMINO_ACIDS, GENETIC_CODE, HUMAN_CODON_PER_THOUSAND, KYTE_DOOLITTLE
from compbio.features import (
    clean_protein,
    codon_entropy,
    effective_number_of_codons,
    genomic_feature_names,
    genomic_feature_vector,
    gc_content,
    gravy,
    isoelectric_point,
    longest_orf_fraction,
    proteomic_feature_names,
    proteomic_feature_vector,
    stop_density,
    translate,
)
from compbio.io import read_fasta, write_fasta
from compbio.simulate import (
    expected_gravy,
    make_genomic_dataset,
    protein_class_profiles,
    simulate_coding,
    simulate_noncoding,
)


def test_genetic_code_and_codon_table_match() -> None:
    assert len(GENETIC_CODE) == 64
    assert set(GENETIC_CODE) == set(HUMAN_CODON_PER_THOUSAND)
    amino_acids = {value for value in GENETIC_CODE.values() if value != "*"}
    assert amino_acids == set(AMINO_ACIDS)
    leucine = ["TTA", "TTG", "CTT", "CTC", "CTA", "CTG"]
    assert HUMAN_CODON_PER_THOUSAND["CTG"] == max(HUMAN_CODON_PER_THOUSAND[codon] for codon in leucine)


def test_translate_stops_at_first_stop() -> None:
    assert translate("ATGGTTTAA") == "MV"
    assert translate("ttt") == "F"


def test_gc_content_counts_only_g_and_c() -> None:
    assert gc_content("GGCC") == 1.0
    assert gc_content("AATT") == 0.0
    assert gc_content("ATGC") == 0.5


def test_gravy_ranks_leucine_above_aspartate() -> None:
    assert gravy("L" * 20) == KYTE_DOOLITTLE["L"]
    assert gravy("L" * 20) > gravy("D" * 20)


def test_isoelectric_point_separates_basic_and_acidic_peptides() -> None:
    assert isoelectric_point("K" * 20) > 9
    assert isoelectric_point("D" * 20) < 5


def test_clean_coding_sequence_is_an_open_reading_frame() -> None:
    rng = np.random.default_rng(1)
    sequence = simulate_coding(rng, 50, frameshift_rate=0.0)
    assert len(sequence) == 150
    assert stop_density(sequence, 0) == 0.0
    assert longest_orf_fraction(sequence) == 1.0
    assert translate(sequence) == clean_protein(translate(sequence))


def test_frameshift_reduces_open_reading_frame_coverage() -> None:
    rng = np.random.default_rng(2)
    clean = [
        longest_orf_fraction(simulate_coding(rng, 80, frameshift_rate=0.0))
        for _ in range(15)
    ]
    shifted = [
        longest_orf_fraction(simulate_coding(rng, 80, frameshift_rate=1.0))
        for _ in range(15)
    ]
    assert np.mean(clean) == 1.0
    assert np.mean(shifted) < 0.95


def test_effective_number_of_codons_detects_bias() -> None:
    biased = effective_number_of_codons("TTT" * 30)
    mixed = effective_number_of_codons("TTT" * 10 + "TTC" * 10)
    assert biased == 20.0
    assert mixed == np.float64(40.0) or abs(mixed - 40.0) < 1e-9
    assert codon_entropy("TTT" * 30) < codon_entropy("TTT" * 10 + "TTC" * 10)


def test_feature_names_match_vectors() -> None:
    dna = simulate_coding(np.random.default_rng(3), 40, frameshift_rate=0.0)
    protein = "ACDEFGHIKLMNPQRSTVWY"
    dna_vector = genomic_feature_vector(dna)
    protein_vector = proteomic_feature_vector(protein)
    assert dna_vector.shape == (len(genomic_feature_names()),)
    assert protein_vector.shape == (len(proteomic_feature_names()),)
    assert dna_vector[genomic_feature_names().index("gc")] == gc_content(dna)
    gravy_index = proteomic_feature_names().index("gravy")
    assert protein_vector[gravy_index] == gravy(protein)


def test_noncoding_gc_matches_the_codon_model() -> None:
    from compbio.simulate import expected_coding_gc

    target = expected_coding_gc()
    rng = np.random.default_rng(4)
    sequences = [simulate_noncoding(rng, 300, embedded_orf_rate=0.0) for _ in range(80)]
    mean_gc = float(np.mean([gc_content(sequence) for sequence in sequences]))
    assert abs(mean_gc - target) < 0.025


def test_membrane_profile_is_more_hydrophobic_than_disordered() -> None:
    profiles = protein_class_profiles()
    assert expected_gravy(profiles["membrane"]) > expected_gravy(profiles["globular"])
    assert expected_gravy(profiles["globular"]) > expected_gravy(profiles["disordered"])


def test_fasta_roundtrip(tmp_path) -> None:
    path = tmp_path / "sample.fasta"
    records = [("seq1", "coding", "ATGGTTTAA"), ("seq2", "globular", "ACDE")]
    write_fasta(path, records)
    assert read_fasta(path) == [("seq1", "coding", "ATGGTTTAA"), ("seq2", "globular", "ACDE")]


def test_models_beat_the_majority_baseline(tmp_path) -> None:
    from compbio.analysis import run_analysis

    summary = run_analysis(
        seed=11,
        n_genomic=80,
        n_proteomic=60,
        out_dir=tmp_path / "results",
        data_dir=tmp_path / "data",
        cv_folds=2,
        importance_repeats=2,
        n_estimators=40,
        make_plots=True,
    )
    genomic = summary["genomic"]["supervised"]
    proteomic = summary["proteomic"]["supervised"]
    assert genomic["models"]["random_forest"]["accuracy"] > genomic["baseline_accuracy"]
    assert genomic["models"]["random_forest"]["roc_auc"] > 0.75
    assert proteomic["models"]["logistic_regression"]["macro_f1"] > proteomic["baseline_accuracy"]
    assert (tmp_path / "results" / "genomic_roc.png").exists()
    assert (tmp_path / "results" / "proteomic_confusion.png").exists()
    assert (tmp_path / "results" / "REPORT.txt").exists()


def test_simulated_dataset_labels_are_balanced() -> None:
    records = make_genomic_dataset(12, np.random.default_rng(5))
    labels = [label for _, label, _ in records]
    assert labels.count("coding") == 12
    assert labels.count("noncoding") == 12
    assert len({sequence for _, _, sequence in records}) == 24
