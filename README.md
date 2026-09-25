# Computational genomic and proteomic analysis

This project applies sequence statistics and machine learning to two biological
classification tasks:

- Genomic: decide whether a DNA window is coding sequence.
- Proteomic: decide whether a protein sequence looks globular, membrane-like, or disordered.

The sequences are simulated from published compositional patterns (human codon
usage, hydrophobicity, and disorder-associated residues). Nothing here is a
clinical classifier, and the code does not design or produce biological material.

## Methods

Genomic features include GC content by codon position, AT and GC skew, stop-codon
density in three frames, longest open-reading-frame coverage, codon entropy,
Wright's effective number of codons, CpG observed/expected, and sense-codon
frequencies.

Proteomic features include amino-acid composition, Kyte-Doolittle GRAVY,
aromaticity, residue mass, a computed isoelectric point, charge at pH 7, and
Chou-Fasman helix, sheet, and turn propensities.

Each task trains an L2 logistic regression and a random forest on a stratified
70/30 split, with stratified cross-validation on the training set. Two checks
sit beside the classifiers:

- An ablation that removes the most direct class signal (reading-frame features
  for DNA; raw amino-acid frequencies for proteins).
- K-means on the same features, scored with the adjusted Rand index.

Noncoding DNA is generated with the same expected GC as the codon model, so
overall GC is not a shortcut. A fraction of coding sequences are frameshifted,
and a fraction of noncoding sequences contain a short open reading frame.

## Run

From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
compbio-analyze
```

`compbio-analyze` writes `data/genomic.fasta`, `data/proteomic.fasta`,
`results/REPORT.txt`, `results/summary.json`, and the figures in `results/`.
The default seed is 7, with 500 coding and 500 noncoding windows and 400
sequences in each protein class.

Useful options:

```bash
compbio-analyze --seed 7 --n-genomic 500 --n-proteomic 400 --out results
compbio-analyze --genomic data/genomic.fasta --proteomic data/proteomic.fasta
```

A labeled FASTA header looks like `>sequence_id|label`. DNA labels in the
simulated file are `coding` and `noncoding`. Protein labels are `globular`,
`membrane`, and `disordered`.

## Figures

- `results/genomic_gc.png` — GC fraction overall and at codon positions 1–3
- `results/genomic_orf.png` — open-reading-frame coverage by class
- `results/genomic_roc.png` — test ROC, including the no-reading-frame ablation
- `results/genomic_confusion.png` — random-forest test confusion matrix
- `results/genomic_importance.png` — permutation importance
- `results/genomic_pca.png` — test sequences in the training principal components
- `results/proteomic_properties.png` — GRAVY and isoelectric point by class
- `results/proteomic_roc.png` — one-versus-rest test ROC
- `results/proteomic_confusion.png` — random-forest test confusion matrix
- `results/proteomic_importance.png` — permutation importance
- `results/proteomic_pca.png` — test sequences in the training principal components

## Scope

Accuracy on this data measures recovery of the simulation, not performance on
RefSeq, a clinical sample, or a pathogen genome. Replace the FASTA files to
run the same features on other labeled sequences of your own.
