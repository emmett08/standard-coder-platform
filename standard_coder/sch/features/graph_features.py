from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Iterable, Sequence

import networkx as nx
import numpy as np

from standard_coder.sch.domain.entities import Commit
from standard_coder.sch.interfaces import ChangeRepModel


def _safe_parse(code: str) -> ast.AST | None:
    try:
        return ast.parse(code)
    except SyntaxError:
        return None


def _extract_import_edges(module_name: str, tree: ast.AST) -> list[tuple[str, str]]:
    edges: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                edges.append((module_name, alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                edges.append((module_name, node.module))
    return edges


def _extract_call_edges(module_name: str, tree: ast.AST) -> list[tuple[str, str]]:
    """Very lightweight call graph.

    Nodes are function names qualified by module path.
    Calls are only recorded when the called function is a simple Name.
    """
    edges: list[tuple[str, str]] = []
    current_func: list[str] = []

    class Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            current_func.append(f"{module_name}:{node.name}")
            self.generic_visit(node)
            current_func.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            current_func.append(f"{module_name}:{node.name}")
            self.generic_visit(node)
            current_func.pop()

        def visit_Call(self, node: ast.Call) -> None:
            if current_func:
                src = current_func[-1]
                if isinstance(node.func, ast.Name):
                    dst = f"{module_name}:{node.func.id}"
                    edges.append((src, dst))
            self.generic_visit(node)

    Visitor().visit(tree)
    return edges


def _build_graph(
    edges: Iterable[tuple[str, str]],
) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_edges_from(edges)
    return g


def _delta_features(before: nx.DiGraph, after: nx.DiGraph) -> dict[str, float]:
    before_nodes = set(before.nodes)
    after_nodes = set(after.nodes)
    before_edges = set(before.edges)
    after_edges = set(after.edges)

    edges_added = after_edges - before_edges
    edges_removed = before_edges - after_edges

    nodes_added = after_nodes - before_nodes
    nodes_removed = before_nodes - after_nodes

    return {
        "nodes_before": float(len(before_nodes)),
        "edges_before": float(len(before_edges)),
        "nodes_after": float(len(after_nodes)),
        "edges_after": float(len(after_edges)),
        "edges_added": float(len(edges_added)),
        "edges_removed": float(len(edges_removed)),
        "nodes_added": float(len(nodes_added)),
        "nodes_removed": float(len(nodes_removed)),
    }


@dataclass
class GraphDeltaChangeRep(ChangeRepModel):
    """Upgrade A (part): dependency + call graph delta features for Python."""

    def fit(self, commits: Sequence[Commit]) -> None:
        return

    def transform(self, commits: Sequence[Commit]) -> np.ndarray:
        rows: list[np.ndarray] = []
        for c in commits:
            if c.language.lower() != "python":
                rows.append(np.zeros(self.feature_dim, dtype=np.float32))
                continue

            before = c.before_blobs or {}
            after = c.after_blobs or {}

            dep_edges_before: list[tuple[str, str]] = []
            dep_edges_after: list[tuple[str, str]] = []
            call_edges_before: list[tuple[str, str]] = []
            call_edges_after: list[tuple[str, str]] = []

            file_paths = set(before.keys()) | set(after.keys())
            for path in file_paths:
                mod = path.replace("/", ".").rsplit(".", 1)[0]
                b_tree = _safe_parse(before.get(path, ""))
                a_tree = _safe_parse(after.get(path, ""))

                if b_tree is not None:
                    dep_edges_before.extend(_extract_import_edges(mod, b_tree))
                    call_edges_before.extend(_extract_call_edges(mod, b_tree))
                if a_tree is not None:
                    dep_edges_after.extend(_extract_import_edges(mod, a_tree))
                    call_edges_after.extend(_extract_call_edges(mod, a_tree))

            dep_before = _build_graph(dep_edges_before)
            dep_after = _build_graph(dep_edges_after)
            call_before = _build_graph(call_edges_before)
            call_after = _build_graph(call_edges_after)

            dep = _delta_features(dep_before, dep_after)
            call = _delta_features(call_before, call_after)

            feats = np.array(
                [
                    dep["nodes_before"],
                    dep["edges_before"],
                    dep["nodes_after"],
                    dep["edges_after"],
                    dep["edges_added"],
                    dep["edges_removed"],
                    dep["nodes_added"],
                    dep["nodes_removed"],
                    call["nodes_before"],
                    call["edges_before"],
                    call["nodes_after"],
                    call["edges_after"],
                    call["edges_added"],
                    call["edges_removed"],
                    call["nodes_added"],
                    call["nodes_removed"],
                ],
                dtype=np.float32,
            )

            rows.append(feats)

        return np.vstack(rows)

    @property
    def feature_dim(self) -> int:
        return 16
