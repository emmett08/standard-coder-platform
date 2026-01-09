from __future__ import annotations

import ast
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Sequence

import numpy as np

from standard_coder.sch.domain.entities import Commit
from standard_coder.sch.interfaces import ChangeRepModel


def _node_type_sequence(tree: ast.AST) -> list[str]:
    seq: list[str] = []
    for node in ast.walk(tree):
        seq.append(type(node).__name__)
    return seq


def _count_node_types(tree: ast.AST) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node in ast.walk(tree):
        name = type(node).__name__
        counts[name] = counts.get(name, 0) + 1
    return counts


def _safe_parse(code: str) -> ast.AST | None:
    try:
        return ast.parse(code)
    except SyntaxError:
        return None


def _opcode_counts(before: list[str], after: list[str]) -> dict[str, int]:
    sm = SequenceMatcher(a=before, b=after)
    counts = {"equal": 0, "insert": 0, "delete": 0, "replace": 0}
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        counts[tag] += (i2 - i1) + (j2 - j1)
    return counts


@dataclass
class AstEditChangeRep(ChangeRepModel):
    """Upgrade A (part): AST edit features for Python commits.

    Notes:
    - This implementation uses Python's `ast` module and therefore
      supports Python only.
    - For other languages it returns a zero vector of the same size.
    """

    node_types: tuple[str, ...] = (
        "FunctionDef",
        "AsyncFunctionDef",
        "ClassDef",
        "If",
        "For",
        "While",
        "Try",
        "With",
        "Call",
        "Assign",
        "Return",
        "Import",
        "ImportFrom",
    )

    def fit(self, commits: Sequence[Commit]) -> None:
        # Fixed feature space; nothing to fit.
        return

    def transform(self, commits: Sequence[Commit]) -> np.ndarray:
        rows: list[np.ndarray] = []
        for c in commits:
            if c.language.lower() != "python":
                rows.append(np.zeros(self.feature_dim, dtype=np.float32))
                continue

            before = c.before_blobs or {}
            after = c.after_blobs or {}

            feats = np.zeros(self.feature_dim, dtype=np.float32)

            type_offset = 0
            opcode_offset = len(self.node_types) * 3

            # Aggregate across changed files.
            counts_before_total = {t: 0 for t in self.node_types}
            counts_after_total = {t: 0 for t in self.node_types}
            counts_absdiff_total = {t: 0 for t in self.node_types}
            opcode_totals = {"equal": 0, "insert": 0, "delete": 0, "replace": 0}

            file_paths = set(before.keys()) | set(after.keys())
            for path in file_paths:
                b_code = before.get(path, "")
                a_code = after.get(path, "")

                b_tree = _safe_parse(b_code)
                a_tree = _safe_parse(a_code)

                if b_tree is not None:
                    b_counts = _count_node_types(b_tree)
                else:
                    b_counts = {}

                if a_tree is not None:
                    a_counts = _count_node_types(a_tree)
                else:
                    a_counts = {}

                for t in self.node_types:
                    b = int(b_counts.get(t, 0))
                    a = int(a_counts.get(t, 0))
                    counts_before_total[t] += b
                    counts_after_total[t] += a
                    counts_absdiff_total[t] += abs(a - b)

                if b_tree is not None and a_tree is not None:
                    b_seq = _node_type_sequence(b_tree)
                    a_seq = _node_type_sequence(a_tree)
                    op = _opcode_counts(b_seq, a_seq)
                    for k in opcode_totals:
                        opcode_totals[k] += int(op[k])

            # Fill per-node-type features: before, after, absdiff.
            for i, t in enumerate(self.node_types):
                feats[type_offset + i] = float(counts_before_total[t])
                feats[type_offset + len(self.node_types) + i] = float(
                    counts_after_total[t]
                )
                feats[type_offset + 2 * len(self.node_types) + i] = float(
                    counts_absdiff_total[t]
                )

            # Opcode features (edit-style proxy).
            feats[opcode_offset + 0] = float(opcode_totals["insert"])
            feats[opcode_offset + 1] = float(opcode_totals["delete"])
            feats[opcode_offset + 2] = float(opcode_totals["replace"])
            feats[opcode_offset + 3] = float(opcode_totals["equal"])

            rows.append(feats)

        return np.vstack(rows).astype(np.float32)

    @property
    def feature_dim(self) -> int:
        return len(self.node_types) * 3 + 4
