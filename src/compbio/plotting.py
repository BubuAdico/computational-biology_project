"""Figures for the genomic and proteomic analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

CODING_COLOR = "#0072B2"
NONCODING_COLOR = "#D55E00"
CLASS_COLORS = {
    "globular": "#0072B2",
    "membrane": "#D55E00",
    "disordered": "#009E73",
    "coding": CODING_COLOR,
    "noncoding": NONCODING_COLOR,
}
CURVE_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")


def _finish(fig: plt.Figure, path: Path, caption: str) -> None:
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.text(0.01, 0.012, caption, fontsize=8, color="#444444")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _color(label: str, fallback_index: int) -> str:
    if label in CLASS_COLORS:
        return CLASS_COLORS[label]
    return CURVE_COLORS[fallback_index % len(CURVE_COLORS)]


def save_grouped_bars(
    path: Path,
    categories: list[str],
    series: dict[str, list[float]],
    errors: dict[str, list[float]] | None,
    xlabel: str,
    ylabel: str,
    title: str,
    caption: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    positions = np.arange(len(categories))
    width = 0.8 / max(len(series), 1)
    for index, (name, values) in enumerate(series.items()):
        offset = (index - (len(series) - 1) / 2) * width
        yerr = None if errors is None else errors[name]
        ax.bar(
            positions + offset,
            values,
            width=width * 0.92,
            label=name,
            color=_color(name.lower(), index),
            yerr=yerr,
            capsize=3,
            error_kw={"elinewidth": 0.8, "ecolor": "#333333"},
        )
    ax.set_xticks(positions)
    ax.set_xticklabels(categories)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.set_ylim(bottom=0)
    _finish(fig, path, caption)


def save_class_bars(
    path: Path,
    labels: list[str],
    values: list[float],
    errors: list[float],
    xlabel: str,
    ylabel: str,
    title: str,
    caption: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    colors = [_color(label, index) for index, label in enumerate(labels)]
    ax.bar(labels, values, color=colors, yerr=errors, capsize=3, error_kw={"elinewidth": 0.8, "ecolor": "#333333"})
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(bottom=min(0.0, min(values) - max(errors) - 0.05))
    _finish(fig, path, caption)


def save_property_panels(
    path: Path,
    labels: list[str],
    panels: list[dict[str, list[float] | str]],
    caption: str,
) -> None:
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.4))
    if len(panels) == 1:
        axes = [axes]
    for axis, panel in zip(axes, panels):
        names = list(labels)
        values = list(panel["values"])  # type: ignore[arg-type]
        errors = list(panel["errors"])  # type: ignore[arg-type]
        colors = [_color(label, index) for index, label in enumerate(names)]
        axis.bar(names, values, color=colors, yerr=errors, capsize=3, error_kw={"elinewidth": 0.8, "ecolor": "#333333"})
        axis.set_xlabel("Protein class")
        axis.set_ylabel(str(panel["ylabel"]))
        axis.set_title(str(panel["title"]))
    fig.suptitle("Physicochemical summaries by protein class", fontsize=13)
    _finish(fig, path, caption)


def save_roc(
    path: Path,
    curves: list[dict[str, object]],
    title: str,
    caption: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#888888", linewidth=1, label="Chance")
    for index, curve in enumerate(curves):
        fpr = np.asarray(curve["fpr"], dtype=float)
        tpr = np.asarray(curve["tpr"], dtype=float)
        label = f"{curve['name']} (AUC {float(curve['auc']):.3f})"
        ax.plot(fpr, tpr, color=CURVE_COLORS[index % len(CURVE_COLORS)], linewidth=1.8, label=label)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    _finish(fig, path, caption)


def roc_points(y_true: np.ndarray, scores: np.ndarray, positive: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    if positive is None:
        fpr, tpr, _ = roc_curve(y_true, scores)
    else:
        fpr, tpr, _ = roc_curve(y_true, scores, pos_label=positive)
    return fpr, tpr


def save_confusion(
    path: Path,
    matrix: np.ndarray,
    labels: list[str],
    title: str,
    caption: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    threshold = float(matrix.max()) / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = int(matrix[row, column])
            ax.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=11,
            )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    _finish(fig, path, caption)


def save_importance(
    path: Path,
    names: list[str],
    means: list[float],
    stds: list[float],
    title: str,
    caption: str,
) -> None:
    order = np.argsort(means)
    ordered_names = [names[index] for index in order]
    ordered_means = [means[index] for index in order]
    ordered_stds = [stds[index] for index in order]
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    positions = np.arange(len(ordered_names))
    ax.barh(positions, ordered_means, xerr=ordered_stds, color=CODING_COLOR, capsize=2, error_kw={"elinewidth": 0.8})
    ax.set_yticks(positions)
    ax.set_yticklabels(ordered_names, fontsize=8)
    ax.set_xlabel("Decrease in macro F1")
    ax.set_title(title)
    _finish(fig, path, caption)


def save_pca(
    path: Path,
    coordinates: np.ndarray,
    labels: np.ndarray,
    variance: np.ndarray,
    title: str,
    caption: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    unique = list(dict.fromkeys(labels.tolist()))
    for index, label in enumerate(unique):
        mask = labels == label
        ax.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=16,
            alpha=0.8,
            color=_color(str(label), index),
            label=str(label),
            linewidths=0,
        )
    ax.set_xlabel(f"PC1 ({variance[0] * 100:.1f}% of training variance)")
    ax.set_ylabel(f"PC2 ({variance[1] * 100:.1f}% of training variance)")
    ax.set_title(title)
    ax.legend(frameon=False, title="Class")
    _finish(fig, path, caption)
