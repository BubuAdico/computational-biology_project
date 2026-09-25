"""FASTA reading and writing for labeled sequences.

Header format: >sequence_id|label
"""

from __future__ import annotations

from pathlib import Path

Record = tuple[str, str, str]


def write_fasta(path: Path, records: list[Record], width: int = 60) -> None:
    lines: list[str] = []
    for sequence_id, label, sequence in records:
        lines.append(f">{sequence_id}|{label}")
        for start in range(0, len(sequence), width):
            lines.append(sequence[start : start + width])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_fasta(path: Path) -> list[Record]:
    records: list[Record] = []
    header: str | None = None
    chunks: list[str] = []

    def flush() -> None:
        if header is None:
            return
        if "|" not in header:
            raise ValueError(f"FASTA header {header!r} must look like >id|label")
        sequence_id, label = header.split("|", 1)
        sequence = "".join(chunks).replace(" ", "")
        if not sequence_id or not label or not sequence:
            raise ValueError(f"Incomplete FASTA record for header {header!r}")
        records.append((sequence_id, label, sequence))

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            flush()
            header = line[1:].strip()
            chunks = []
        else:
            if header is None:
                raise ValueError("FASTA sequence appears before a header")
            chunks.append(line)
    flush()
    if not records:
        raise ValueError(f"No sequences found in {path}")
    return records
