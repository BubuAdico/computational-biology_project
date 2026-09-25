"""Train genomic and proteomic sequence classifiers and write a report."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from compbio import __version__
from compbio.features import (
    GENOMIC_STRUCTURE_FEATURES,
    clean_dna,
    clean_protein,
    genomic_feature_names,
    genomic_feature_vector,
    is_composition_feature,
    proteomic_feature_names,
    proteomic_feature_vector,
)
from compbio.io import Record, read_fasta, write_fasta
from compbio.models import cluster_ari, coding_coefficients, evaluate_models, split_indices
from compbio.plotting import (
    roc_points,
    save_class_bars,
    save_confusion,
    save_grouped_bars,
    save_importance,
    save_pca,
    save_property_panels,
    save_roc,
)
from compbio.simulate import (
    DIRICHLET_CONCENTRATION,
    EMBEDDED_ORF_RATE,
    FRAMESHIFT_RATE,
    expected_coding_gc,
    make_genomic_dataset,
    make_proteomic_dataset,
)

GENOMIC_STATS = (
    "gc",
    "gc1",
    "gc2",
    "gc3",
    "longest_orf_fraction",
    "stop_density_f0",
    "codon_entropy",
    "effective_number_of_codons",
    "cpg_oe",
)
PROTEOMIC_STATS = (
    "gravy",
    "isoelectric_point",
    "charge_ph7",
    "helix_propensity",
    "sheet_propensity",
    "turn_propensity",
    "aromaticity",
)


def run_analysis(
    seed: int = 7,
    n_genomic: int = 500,
    n_proteomic: int = 400,
    out_dir: Path = Path("results"),
    data_dir: Path = Path("data"),
    genomic_path: Path | None = None,
    proteomic_path: Path | None = None,
    cv_folds: int = 5,
    importance_repeats: int = 8,
    n_estimators: int = 200,
    make_plots: bool = True,
) -> dict[str, Any]:
    """Simulate or load sequences, fit models, and write metrics and figures."""
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    if genomic_path is None:
        genomic_records = make_genomic_dataset(n_genomic, rng)
        genomic_path = data_dir / "genomic.fasta"
        write_fasta(genomic_path, genomic_records)
    else:
        genomic_records = read_fasta(genomic_path)

    if proteomic_path is None:
        proteomic_records = make_proteomic_dataset(n_proteomic, rng)
        proteomic_path = data_dir / "proteomic.fasta"
        write_fasta(proteomic_path, proteomic_records)
    else:
        proteomic_records = read_fasta(proteomic_path)

    _validate(genomic_records, clean_dna, "Genomic FASTA")
    _validate(proteomic_records, clean_protein, "Proteomic FASTA")

    genomic = _run_task(
        records=genomic_records,
        names=genomic_feature_names(),
        vectorizer=genomic_feature_vector,
        seed=seed,
        cv_folds=cv_folds,
        importance_repeats=importance_repeats,
        n_estimators=n_estimators,
        positive_label="coding",
        structure_features=GENOMIC_STRUCTURE_FEATURES,
        task="genomic",
    )
    proteomic = _run_task(
        records=proteomic_records,
        names=proteomic_feature_names(),
        vectorizer=proteomic_feature_vector,
        seed=seed + 1,
        cv_folds=cv_folds,
        importance_repeats=importance_repeats,
        n_estimators=n_estimators,
        positive_label=None,
        structure_features=None,
        task="proteomic",
    )

    summary: dict[str, Any] = {
        "version": __version__,
        "seed": seed,
        "simulation": {
            "frameshift_rate": FRAMESHIFT_RATE,
            "embedded_orf_rate": EMBEDDED_ORF_RATE,
            "dirichlet_concentration": DIRICHLET_CONCENTRATION,
            "expected_coding_gc": expected_coding_gc(),
            "genomic_source": str(genomic_path),
            "proteomic_source": str(proteomic_path),
        },
        "genomic": genomic["public"],
        "proteomic": proteomic["public"],
    }
    summary = _sanitize(summary)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = _render_report(summary)
    (out_dir / "REPORT.txt").write_text(report, encoding="utf-8")

    if make_plots:
        _plot_genomic(out_dir, genomic, seed)
        _plot_proteomic(out_dir, proteomic, seed)
    print(report)
    print(f"\nWrote {out_dir / 'REPORT.txt'} and {out_dir / 'summary.json'}")
    return summary


def _validate(records: list[Record], cleaner: Callable[[str], str], task_name: str) -> None:
    counts = Counter(label for _, label, _ in records)
    if len(counts) < 2:
        raise ValueError(f"{task_name} needs at least two labels, found {dict(counts)}")
    if any(count < 8 for count in counts.values()):
        raise ValueError(f"{task_name} needs at least 8 sequences in every class, found {dict(counts)}")
    for sequence_id, _, sequence in records:
        try:
            cleaner(sequence)
        except ValueError as exc:
            raise ValueError(f"{task_name} record {sequence_id} is invalid: {exc}") from exc


def _matrix(records: list[Record], vectorizer: Callable[[str], np.ndarray]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    features = np.vstack([vectorizer(sequence) for _, _, sequence in records])
    labels = np.array([label for _, label, _ in records])
    identities = [sequence_id for sequence_id, _, _ in records]
    return features, labels, identities


def _column_stats(features: np.ndarray, labels: np.ndarray, names: list[str], wanted: tuple[str, ...]) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for name in wanted:
        column = features[:, names.index(name)]
        stats[name] = {}
        for label in sorted(set(labels.tolist())):
            values = column[labels == label]
            stats[name][label] = _mean_std(values)
    return stats


def _summed_column_stats(features: np.ndarray, labels: np.ndarray, names: list[str], columns: str, key: str) -> dict[str, Any]:
    indices = [names.index(f"aa_{residue}") for residue in columns]
    values = features[:, indices].sum(axis=1)
    return {key: {label: _mean_std(values[labels == label]) for label in sorted(set(labels.tolist()))}}


def _mean_std(values: np.ndarray) -> dict[str, float | int]:
    std = float(values.std(ddof=1)) if values.size > 1 else 0.0
    return {"mean": float(values.mean()), "std": std, "n": int(values.size)}


def _public_metrics(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "baseline_accuracy": result["baseline_accuracy"],
        "n_train": int(len(result["y_train"])),
        "n_test": int(len(result["y_test"])),
        "models": result["metrics"],
    }


def _run_task(
    records: list[Record],
    names: list[str],
    vectorizer: Callable[[str], np.ndarray],
    seed: int,
    cv_folds: int,
    importance_repeats: int,
    n_estimators: int,
    positive_label: str | None,
    structure_features: frozenset[str] | None,
    task: str,
) -> dict[str, Any]:
    features, labels, _ = _matrix(records, vectorizer)
    train_idx, test_idx = split_indices(labels, seed)
    full = evaluate_models(
        features[train_idx],
        features[test_idx],
        labels[train_idx],
        labels[test_idx],
        seed,
        cv_folds=cv_folds,
        n_estimators=n_estimators,
        positive_label=positive_label,
    )
    if structure_features is not None:
        keep = [index for index, name in enumerate(names) if name not in structure_features]
        ablation_name = "without_reading_frame_features"
    else:
        keep = [index for index, name in enumerate(names) if not is_composition_feature(name)]
        ablation_name = "physicochemical_features_only"
    ablation = evaluate_models(
        features[train_idx][:, keep],
        features[test_idx][:, keep],
        labels[train_idx],
        labels[test_idx],
        seed,
        cv_folds=cv_folds,
        n_estimators=n_estimators,
        positive_label=positive_label,
    )
    n_clusters = len(set(labels.tolist()))
    ari = cluster_ari(
        features[train_idx],
        labels[train_idx],
        features[test_idx],
        labels[test_idx],
        n_clusters,
        seed,
    )
    ranked: list[dict[str, float | str]] = []
    if importance_repeats > 0:
        forest = full["models"]["random_forest"]
        importance = permutation_importance(
            forest,
            features[test_idx],
            labels[test_idx],
            n_repeats=importance_repeats,
            random_state=seed,
            scoring="f1_macro",
            n_jobs=-1,
        )
        order = np.argsort(importance.importances_mean)[::-1]
        ranked = [
            {
                "feature": names[int(index)],
                "mean_decrease_f1": float(importance.importances_mean[int(index)]),
                "std": float(importance.importances_std[int(index)]),
            }
            for index in order
        ]

    if task == "genomic":
        descriptive = _column_stats(features, labels, names, GENOMIC_STATS)
        coefficients = coding_coefficients(full["models"]["logistic_regression"], names)
    else:
        descriptive = _column_stats(features, labels, names, PROTEOMIC_STATS)
        descriptive.update(_summed_column_stats(features, labels, names, "LIVF", "hydrophobic_fraction"))
        descriptive.update(_summed_column_stats(features, labels, names, "PESQK", "disorder_residue_fraction"))
        coefficients = []

    class_counts = dict(Counter(labels.tolist()))
    public = {
        "class_counts": class_counts,
        "n_features": len(names),
        "descriptive": descriptive,
        "supervised": _public_metrics(full),
        "ablation": {
            "name": ablation_name,
            "n_features": len(keep),
            "supervised": _public_metrics(ablation),
        },
        "kmeans_test_adjusted_rand": ari,
        "permutation_importance": ranked[:15],
        "logistic_coefficients": coefficients[:12],
    }
    return {
        "public": public,
        "features": features,
        "labels": labels,
        "names": names,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "full": full,
        "ablation": ablation,
        "positive_label": positive_label,
    }


def _plot_genomic(out_dir: Path, task: dict[str, Any], seed: int) -> None:
    descriptive = task["public"]["descriptive"]
    labels_present = sorted({label for feature in descriptive.values() for label in feature})
    n_total = int(sum(task["public"]["class_counts"].values()))
    caption = f"Source: simulated DNA windows, seed {seed}, n = {n_total}. Error bars are standard deviations."
    categories = ["Overall GC", "GC1", "GC2", "GC3"]
    keys = ["gc", "gc1", "gc2", "gc3"]
    series = {
        label.capitalize(): [descriptive[key][label]["mean"] for key in keys]
        for label in ("coding", "noncoding")
        if label in labels_present
    }
    errors = {
        label.capitalize(): [descriptive[key][label]["std"] for key in keys]
        for label in ("coding", "noncoding")
        if label in labels_present
    }
    save_grouped_bars(
        out_dir / "genomic_gc.png",
        categories,
        series,
        errors,
        xlabel="Base-composition statistic",
        ylabel="GC fraction",
        title="GC content by codon position",
        caption=caption,
    )
    save_class_bars(
        out_dir / "genomic_orf.png",
        ["coding", "noncoding"],
        [descriptive["longest_orf_fraction"][label]["mean"] for label in ("coding", "noncoding")],
        [descriptive["longest_orf_fraction"][label]["std"] for label in ("coding", "noncoding")],
        xlabel="Sequence class",
        ylabel="Longest open reading frame / length",
        title="Open reading frame coverage",
        caption=caption,
    )
    _plot_roc_and_model_figures(
        out_dir,
        task,
        prefix="genomic",
        roc_title="Test ROC for coding-sequence classification",
        confusion_title="Test confusion matrix, genomic random forest",
        importance_title="Test permutation importance, genomic random forest",
        pca_title="Genomic feature space, held-out windows",
        seed=seed,
        positive_label="coding",
    )


def _plot_proteomic(out_dir: Path, task: dict[str, Any], seed: int) -> None:
    descriptive = task["public"]["descriptive"]
    class_order = [label for label in ("globular", "membrane", "disordered") if label in descriptive["gravy"]]
    n_total = int(sum(task["public"]["class_counts"].values()))
    caption = f"Source: simulated protein sequences, seed {seed}, n = {n_total}. Error bars are standard deviations."
    save_property_panels(
        out_dir / "proteomic_properties.png",
        class_order,
        [
            {
                "title": "Hydrophobicity",
                "ylabel": "Mean GRAVY (Kyte-Doolittle index)",
                "values": [descriptive["gravy"][label]["mean"] for label in class_order],
                "errors": [descriptive["gravy"][label]["std"] for label in class_order],
            },
            {
                "title": "Isoelectric point",
                "ylabel": "Mean estimated pI (pH)",
                "values": [descriptive["isoelectric_point"][label]["mean"] for label in class_order],
                "errors": [descriptive["isoelectric_point"][label]["std"] for label in class_order],
            },
        ],
        caption,
    )
    _plot_roc_and_model_figures(
        out_dir,
        task,
        prefix="proteomic",
        roc_title="Test ROC for protein-class classification",
        confusion_title="Test confusion matrix, proteomic random forest",
        importance_title="Test permutation importance, proteomic random forest",
        pca_title="Proteomic feature space, held-out sequences",
        seed=seed,
        positive_label=None,
    )


def _plot_roc_and_model_figures(
    out_dir: Path,
    task: dict[str, Any],
    prefix: str,
    roc_title: str,
    confusion_title: str,
    importance_title: str,
    pca_title: str,
    seed: int,
    positive_label: str | None,
) -> None:
    full = task["full"]
    n_test = len(full["y_test"])
    curves: list[dict[str, object]] = []
    if positive_label is not None:
        specs = (
            ("Logistic regression", full, "logistic_regression"),
            ("Random forest", full, "random_forest"),
            ("Logistic regression, no ORF or stop features", task["ablation"], "logistic_regression"),
        )
        for name, result, model_name in specs:
            model = result["models"][model_name]
            classes = list(model.classes_)
            scores = model.predict_proba(result["x_test"])[:, classes.index(positive_label)]
            fpr, tpr = roc_points(result["y_test"], scores, positive_label)
            curves.append(
                {
                    "name": name,
                    "fpr": fpr,
                    "tpr": tpr,
                    "auc": result["metrics"][model_name]["roc_auc"],
                }
            )
        roc_caption = (
            f"Source: held-out test sequences, seed {seed}, n = {n_test}. "
            "The positive class is coding sequence."
        )
    else:
        model = full["models"]["random_forest"]
        classes = list(model.classes_)
        probabilities = model.predict_proba(full["x_test"])
        for label in classes:
            scores = probabilities[:, classes.index(label)]
            binary = (full["y_test"] == label).astype(int)
            fpr, tpr = roc_points(binary, scores)
            curves.append(
                {
                    "name": f"Random forest, {label} vs rest",
                    "fpr": fpr,
                    "tpr": tpr,
                    "auc": float(roc_auc_score(binary, scores)),
                }
            )
        roc_caption = f"Source: held-out test sequences, seed {seed}, n = {n_test}. One-versus-rest curves."

    save_roc(out_dir / f"{prefix}_roc.png", curves, roc_title, roc_caption)
    forest_metrics = full["metrics"]["random_forest"]
    matrix = np.asarray(forest_metrics["confusion_matrix"])
    save_confusion(
        out_dir / f"{prefix}_confusion.png",
        matrix,
        list(forest_metrics["classes"]),
        confusion_title,
        f"Source: held-out test sequences, seed {seed}, n = {n_test}.",
    )
    ranked = task["public"]["permutation_importance"][:12]
    if ranked:
        save_importance(
            out_dir / f"{prefix}_importance.png",
            [str(item["feature"]) for item in reversed(ranked)],
            [float(item["mean_decrease_f1"]) for item in reversed(ranked)],
            [float(item["std"]) for item in reversed(ranked)],
            importance_title,
            f"Source: permutation on the held-out test set, seed {seed}. Error bars are standard deviations across repeats.",
        )
    scaler = StandardScaler().fit(task["features"][task["train_idx"]])
    pca = PCA(n_components=2, random_state=seed)
    pca.fit(scaler.transform(task["features"][task["train_idx"]]))
    coordinates = pca.transform(scaler.transform(task["features"][task["test_idx"]]))
    save_pca(
        out_dir / f"{prefix}_pca.png",
        coordinates,
        task["labels"][task["test_idx"]],
        pca.explained_variance_ratio_,
        pca_title,
        f"Source: principal components fit on the training set only; points are the test set. Seed {seed}.",
    )


def _render_report(summary: dict[str, Any]) -> str:
    genomic = summary["genomic"]
    proteomic = summary["proteomic"]
    simulation = summary["simulation"]
    lines = [
        "Computational genomic and proteomic analysis",
        "============================================",
        "",
        f"Package version {summary['version']}. Random seed {summary['seed']}.",
        "",
        "What this project does",
        "----------------------",
        "Computational sequence statistics are calculated for DNA windows and",
        "protein sequences. Logistic regression and random forests then use those",
        "features to predict whether a DNA window is coding sequence, and which",
        "compositional class a protein belongs to.",
        "",
        "Data",
        "----",
        "All sequences are simulated. Coding DNA is sampled from a human codon-usage "
        f"table (expected GC {simulation['expected_coding_gc']:.3f}). With probability "
        f"{simulation['frameshift_rate']:.2f}, one base is deleted so the reading frame breaks. "
        "Noncoding DNA uses an independent-base model with the same expected GC. "
        f"With probability {simulation['embedded_orf_rate']:.2f}, a short coding stretch is "
        "inserted so open reading frames are not a perfect rule.",
        "",
        "Proteins are sampled from three amino-acid profiles built on the human "
        "background: globular, membrane (hydrophobic residues enriched), and "
        "disordered (proline, glutamate, serine, glutamine, and lysine enriched). "
        "Each sequence draws its own frequencies from a Dirichlet distribution "
        f"(concentration {simulation['dirichlet_concentration']:.0f}).",
        "",
        f"Genomic FASTA: {simulation['genomic_source']}",
        f"Proteomic FASTA: {simulation['proteomic_source']}",
        "",
        "Genomic descriptive statistics",
        "------------------------------",
        "Statistics below are means over the full simulated cohort, before the",
        "train/test split. Standard deviations are in parentheses.",
        "",
        _stat_table(genomic["descriptive"], GENOMIC_STATS),
        "",
        _genomic_story(genomic["descriptive"]),
        "",
        "Genomic classification",
        "----------------------",
        _class_line(genomic["class_counts"]),
        f"Features: {genomic['n_features']}. Training sequences: {genomic['supervised']['n_train']}. "
        f"Test sequences: {genomic['supervised']['n_test']}.",
        f"Majority-class baseline accuracy: {_fmt(genomic['supervised']['baseline_accuracy'])}.",
        "ROC AUC treats coding sequence as the positive class.",
        "",
        _model_table(genomic["supervised"]["models"]),
        "",
        _per_class(genomic["supervised"]["models"]),
        "",
        "Ablation without reading-frame features",
        "---------------------------------------",
        "This model drops longest open-reading-frame coverage and stop-codon density",
        "in all three frames. Codon composition, GC by codon position, skew, CpG",
        "observed/expected, codon entropy, and Wright's effective number of codons remain.",
        f"Features retained: {genomic['ablation']['n_features']}.",
        "",
        _model_table(genomic["ablation"]["supervised"]["models"]),
        "",
        _genomic_ablation_story(genomic),
        "",
        f"K-means (k = 2) adjusted Rand index on the test set: {_fmt(genomic['kmeans_test_adjusted_rand'])}.",
        "An index of 0 matches chance agreement and 1 matches the true labels.",
        "",
        "Largest logistic-regression coefficients for the coding class",
        "(features were standardized on the training set):",
        _coefficient_lines(genomic["logistic_coefficients"]),
        "",
        _coefficient_story(genomic["logistic_coefficients"]),
        "",
        "Largest random-forest permutation importances on the test set",
        "(mean drop in macro F1):",
        _importance_lines(genomic["permutation_importance"]),
        "",
        _importance_story(genomic["permutation_importance"], genomic=True),
        "",
        "Proteomic descriptive statistics",
        "--------------------------------",
        _stat_table(
            proteomic["descriptive"],
            PROTEOMIC_STATS + ("hydrophobic_fraction", "disorder_residue_fraction"),
        ),
        "",
        _proteomic_story(proteomic["descriptive"]),
        "",
        "Proteomic classification",
        "------------------------",
        _class_line(proteomic["class_counts"]),
        f"Features: {proteomic['n_features']}. Training sequences: {proteomic['supervised']['n_train']}. "
        f"Test sequences: {proteomic['supervised']['n_test']}.",
        f"Majority-class baseline accuracy: {_fmt(proteomic['supervised']['baseline_accuracy'])}.",
        "ROC AUC is the macro average of one-versus-rest curves.",
        "",
        _model_table(proteomic["supervised"]["models"]),
        "",
        _per_class(proteomic["supervised"]["models"]),
        "",
        _class_overlap_story(proteomic),
        "",
        "Ablation using physicochemical features only",
        "--------------------------------------------",
        "Amino-acid frequencies and composition entropy are removed. GRAVY,",
        "aromaticity, mean residue mass, isoelectric point, charge at pH 7, and",
        "Chou-Fasman helix, sheet, and turn propensities remain.",
        f"Features retained: {proteomic['ablation']['n_features']}.",
        "",
        _model_table(proteomic["ablation"]["supervised"]["models"]),
        "",
        _proteomic_ablation_story(proteomic),
        "",
        f"K-means (k = 3) adjusted Rand index on the test set: {_fmt(proteomic['kmeans_test_adjusted_rand'])}.",
        "",
        "Largest random-forest permutation importances on the test set:",
        _importance_lines(proteomic["permutation_importance"]),
        "",
        _importance_story(proteomic["permutation_importance"], genomic=False),
        "",
        "How to read the result",
        "----------------------",
        "The held-out accuracy should be compared with the majority-class baseline,",
        "not with a perfect score. The ablations show how much of that accuracy",
        "survives after the most direct class signals are removed. Permutation",
        "importance ranks features for the random forest that was actually tested;",
        "it is not a causal laboratory measurement.",
        "",
        "Scope",
        "-----",
        "This is a methods demonstration on simulated sequences. It is not a",
        "clinical, diagnostic, or pathogen classifier, and it does not design,",
        "express, or produce biological material.",
        "",
    ]
    return "\n".join(lines)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number) or math.isinf(number):
        return "n/a"
    return f"{number:.{digits}f}"


def _per_class(models: dict[str, Any]) -> str:
    report = models["random_forest"]["classification_report"]
    lines = ["Random-forest test precision, recall, and F1 by class:"]
    for label in models["random_forest"]["classes"]:
        row = report[str(label)]
        lines.append(
            f"  {label}: precision {_fmt(row['precision'])}, "
            f"recall {_fmt(row['recall'])}, F1 {_fmt(row['f1-score'])}"
        )
    return "\n".join(lines)


def _class_line(counts: dict[str, int]) -> str:
    parts = [f"{label} {count}" for label, count in sorted(counts.items())]
    return "Class counts: " + ", ".join(parts) + "."


def _stat_table(stats: dict[str, Any], features: tuple[str, ...]) -> str:
    labels = sorted({label for feature in features if feature in stats for label in stats[feature]})
    headers = ["statistic", *labels]
    rows = []
    for feature in features:
        if feature not in stats:
            continue
        rows.append(
            [
                feature,
                *[
                    f"{_fmt(stats[feature][label]['mean'])} ({_fmt(stats[feature][label]['std'])})"
                    for label in labels
                ],
            ]
        )
    return _text_table(headers, rows)


def _model_table(models: dict[str, Any]) -> str:
    headers = ["model", "accuracy", "macro F1", "ROC AUC", "CV accuracy"]
    rows = []
    for name in ("logistic_regression", "random_forest"):
        metrics = models[name]
        cv = _fmt(metrics["cv_accuracy_mean"])
        if metrics["cv_accuracy_std"] is not None and not (
            isinstance(metrics["cv_accuracy_std"], float) and math.isnan(metrics["cv_accuracy_std"])
        ):
            cv = f"{cv} +/- {_fmt(metrics['cv_accuracy_std'])}"
        rows.append(
            [
                name,
                _fmt(metrics["accuracy"]),
                _fmt(metrics["macro_f1"]),
                _fmt(metrics["roc_auc"]),
                cv,
            ]
        )
    return _text_table(headers, rows)


def _text_table(headers: list[str], rows: list[list[str]]) -> str:
    string_rows = [[str(cell) for cell in row] for row in rows]
    widths = [
        max(len(row[column]) for row in [headers, *string_rows])
        for column in range(len(headers))
    ]

    def format_row(row: list[str]) -> str:
        return "  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row))

    divider = "  ".join("-" * width for width in widths)
    return "\n".join([format_row(headers), divider, *[format_row(row) for row in string_rows]])


def _coefficient_lines(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "  (none)"
    return "\n".join(
        f"  {index:2d}. {row['coefficient']:+.3f}  {row['feature']}"
        for index, row in enumerate(rows[:8], start=1)
    )


def _importance_lines(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "  (none)"
    return "\n".join(
        f"  {index:2d}. {_fmt(row['mean_decrease_f1'])} +/- {_fmt(row['std'])}  {row['feature']}"
        for index, row in enumerate(rows[:8], start=1)
    )


def _mean(stats: dict[str, Any], feature: str, label: str) -> float | None:
    try:
        return float(stats[feature][label]["mean"])
    except (KeyError, TypeError):
        return None


def _genomic_story(stats: dict[str, Any]) -> str:
    coding_gc = _mean(stats, "gc", "coding")
    noncoding_gc = _mean(stats, "gc", "noncoding")
    coding_gc3 = _mean(stats, "gc3", "coding")
    noncoding_gc3 = _mean(stats, "gc3", "noncoding")
    coding_gc2 = _mean(stats, "gc2", "coding")
    noncoding_gc2 = _mean(stats, "gc2", "noncoding")
    coding_orf = _mean(stats, "longest_orf_fraction", "coding")
    noncoding_orf = _mean(stats, "longest_orf_fraction", "noncoding")
    coding_cpg = _mean(stats, "cpg_oe", "coding")
    noncoding_cpg = _mean(stats, "cpg_oe", "noncoding")
    required = (
        coding_gc, noncoding_gc, coding_gc2, noncoding_gc2, coding_gc3, noncoding_gc3,
        coding_orf, noncoding_orf, coding_cpg, noncoding_cpg,
    )
    if any(value is None for value in required):
        return "Descriptive comparison uses the coding and noncoding labels."
    assert coding_gc is not None and noncoding_gc is not None
    assert coding_gc2 is not None and noncoding_gc2 is not None
    assert coding_gc3 is not None and noncoding_gc3 is not None
    assert coding_orf is not None and noncoding_orf is not None
    assert coding_cpg is not None and noncoding_cpg is not None
    sentence = (
        f"Mean GC is {coding_gc:.3f} in coding windows and {noncoding_gc:.3f} in noncoding windows. "
        f"Mean second-position GC is {coding_gc2:.3f} versus {noncoding_gc2:.3f}, and "
        f"mean third-position GC is {coding_gc3:.3f} versus {noncoding_gc3:.3f}. "
        f"Mean open-reading-frame coverage is {coding_orf:.3f} versus {noncoding_orf:.3f}. "
        f"Mean CpG observed/expected is {coding_cpg:.3f} versus {noncoding_cpg:.3f}."
    )
    if abs(coding_gc - noncoding_gc) < 0.03 and coding_gc3 - noncoding_gc3 > 0.02:
        sentence += " Overall GC is matched by design; third-position GC still separates the classes."
    if noncoding_gc2 - coding_gc2 > 0.02:
        sentence += " Second-position GC is lower in coding windows, which follows from the structure of the genetic code."
    return sentence


def _genomic_ablation_story(genomic: dict[str, Any]) -> str:
    full = genomic["supervised"]["models"]["logistic_regression"]["roc_auc"]
    reduced = genomic["ablation"]["supervised"]["models"]["logistic_regression"]["roc_auc"]
    if full is None or reduced is None:
        return "Ablation AUC could not be computed."
    drop = float(full) - float(reduced)
    return (
        f"Removing explicit reading-frame features changes logistic-regression ROC AUC "
        f"from {_fmt(full)} to {_fmt(reduced)} (difference {_fmt(drop)}). "
        "A remaining AUC above the 0.5 chance line means codon usage and base composition "
        "still carry coding signal."
    )


def _proteomic_story(stats: dict[str, Any]) -> str:
    gravy = {label: _mean(stats, "gravy", label) for label in ("membrane", "globular", "disordered")}
    hydrophobic = {
        label: _mean(stats, "hydrophobic_fraction", label) for label in ("membrane", "globular", "disordered")
    }
    if any(value is None for value in gravy.values()):
        return "Physicochemical means are reported for the labels present in the FASTA file."
    sentence = (
        "Mean GRAVY is "
        + ", ".join(f"{_fmt(value)} for {label}" for label, value in gravy.items())
        + ". Mean hydrophobic-residue fraction (L, I, V, F) is "
        + ", ".join(f"{_fmt(value)} for {label}" for label, value in hydrophobic.items())
        + "."
    )
    membrane = gravy["membrane"]
    others = [gravy["globular"], gravy["disordered"]]
    if membrane is not None and all(other is not None and membrane > other for other in others):
        sentence += " Membrane-like sequences are the most hydrophobic, which follows their frequency profile."
    return sentence


def _proteomic_ablation_story(proteomic: dict[str, Any]) -> str:
    full = proteomic["supervised"]["models"]["logistic_regression"]["macro_f1"]
    reduced = proteomic["ablation"]["supervised"]["models"]["logistic_regression"]["macro_f1"]
    return (
        f"Physicochemical features alone reach logistic-regression macro F1 {_fmt(reduced)}, "
        f"compared with {_fmt(full)} when amino-acid frequencies are included."
    )


def _coefficient_story(rows: list[dict[str, Any]]) -> str:
    cpg_codons = [
        str(row["feature"])
        for row in rows[:8]
        if "CG" in str(row["feature"]) and float(row["coefficient"]) < 0
    ]
    if len(cpg_codons) < 3:
        return ""
    listed = ", ".join(codon.removeprefix("codon_") for codon in cpg_codons[:4])
    return (
        "Several of the strongest negative coefficients belong to CpG-containing codons "
        f"({listed}). Those codons are rare in the human usage table, so a high count "
        "argues against a coding window."
    )


def _class_overlap_story(proteomic: dict[str, Any]) -> str:
    report = proteomic["supervised"]["models"]["random_forest"]["classification_report"]
    membrane = report.get("membrane", {}).get("f1-score")
    globular = report.get("globular", {}).get("f1-score")
    disordered = report.get("disordered", {}).get("f1-score")
    if membrane is None or globular is None or disordered is None:
        return ""
    if float(membrane) >= 0.99 and min(float(globular), float(disordered)) < 0.98:
        return (
            "The membrane class is separated cleanly. Remaining errors fall between "
            "globular and disordered sequences, whose compositions overlap."
        )
    return ""


def _importance_story(rows: list[dict[str, Any]], genomic: bool) -> str:
    if not rows:
        return "Permutation importance was not computed for this run."
    top = [str(row["feature"]) for row in rows[:8]]
    listed = ", ".join(top[:5])
    if genomic:
        structure = GENOMIC_STRUCTURE_FEATURES.intersection(top)
        codon = [name for name in top if name.startswith("codon_") or name in {"gc3", "codon_entropy", "effective_number_of_codons"}]
        parts = [f"The highest-ranked genomic features include {listed}."]
        if structure:
            parts.append("Reading-frame statistics are in that set.")
        if codon:
            parts.append("Codon-usage statistics are in that set as well.")
        return " ".join(parts)
    hydrophobic = [name for name in top if name in {"gravy", "aa_L", "aa_I", "aa_V", "aa_F", "sheet_propensity"}]
    sentence = f"The highest-ranked proteomic features include {listed}."
    if hydrophobic:
        sentence += " Hydrophobicity-related features appear in that ranking."
    if float(rows[0]["mean_decrease_f1"]) < 0.03:
        sentence += (
            " The drops are small because composition features are redundant: "
            "permuting one leaves the others available to the forest."
        )
    return sentence


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, np.ndarray):
        return _sanitize(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return number
    return value


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-genomic", type=int, default=500, help="Coding and noncoding sequences each")
    parser.add_argument("--n-proteomic", type=int, default=400, help="Sequences in each protein class")
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--genomic", type=Path, default=None, help="Existing labeled FASTA; default is to simulate")
    parser.add_argument("--proteomic", type=Path, default=None, help="Existing labeled FASTA; default is to simulate")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--importance-repeats", type=int, default=8)
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    run_analysis(
        seed=args.seed,
        n_genomic=args.n_genomic,
        n_proteomic=args.n_proteomic,
        out_dir=args.out,
        data_dir=args.data,
        genomic_path=args.genomic,
        proteomic_path=args.proteomic,
        cv_folds=args.cv_folds,
        importance_repeats=args.importance_repeats,
        n_estimators=args.trees,
        make_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
