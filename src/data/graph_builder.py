"""Build fixed-order NumPy graphs with explicit edge provenance.

No real topology is inferred. The default graph is an assumed bidirectional
star. Static attributes encode node type, an engineering-scaled hypothetical
rating and a rating-known mask. Edge weights of one denote membership, not
impedance, distance, transfer capacity or learned similarity. No self loops
are added; a future GNN may explicitly choose its own self-loop policy.
"""

from copy import deepcopy
import math
from pathlib import Path

import numpy as np
import yaml


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs/forecasting_data.yaml"
NODE_TYPES = ("building", "solar", "wind", "bus")


class GraphDataError(ValueError):
    """Invalid graph configuration or unsupported topology claim."""


def load_forecasting_config(path=DEFAULT_CONFIG):
    try:
        config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise GraphDataError(f"Cannot load forecasting data configuration: {exc}") from exc
    if not isinstance(config, dict) or config.get("version") != 1:
        raise GraphDataError("Forecasting data configuration requires version: 1.")
    return config


def build_graph(config=None):
    """Return edge_index, node_attributes and JSON-safe ordered graph metadata.

    Explicit edges can use construction: explicit with edges containing source,
    target, basis (modeling_assumption or documented_physical), and reference.
    The caller is responsible for the truth of physical documentation; it is
    never inferred by this utility. Directions are exactly as configured.
    """
    config = deepcopy(config) if config is not None else load_forecasting_config()
    spec = config["graph"]
    nodes = spec.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise GraphDataError("Graph requires an ordered nonempty node list.")
    ids = [node.get("id") for node in nodes]
    if not all(isinstance(node, str) and node.strip() for node in ids) or len(ids) != len(set(ids)):
        raise GraphDataError("Node IDs must be unique nonempty text.")
    scale = spec.get("capacity_scale_kw")
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise GraphDataError("capacity_scale_kw must be positive and finite.")
    attributes = []
    for node in nodes:
        if node.get("type") not in NODE_TYPES:
            raise GraphDataError(f"Unsupported node type for {node['id']}.")
        capacity = node.get("capacity_kw")
        if capacity is not None and (isinstance(capacity, bool) or not isinstance(capacity, (int, float)) or not math.isfinite(capacity) or capacity <= 0):
            raise GraphDataError(f"Invalid capacity_kw for {node['id']}.")
        attributes.append([float(node["type"] == kind) for kind in NODE_TYPES]
                          + [0 if capacity is None else capacity / scale, float(capacity is not None)])
    positions = {node: index for index, node in enumerate(ids)}
    construction = spec.get("construction")
    if construction == "assumed_shared_bus":
        bus = spec.get("bus_id")
        if bus not in positions or nodes[positions[bus]]["type"] != "bus":
            raise GraphDataError("Assumed shared-bus graph requires a valid bus node.")
        if not isinstance(spec.get("bidirectional"), bool) or not spec.get("reference"):
            raise GraphDataError("Document the assumed graph and its edge directions.")
        edges = []
        for node in ids:
            if node != bus:
                edges.append({"source": node, "target": bus, "basis": "modeling_assumption", "reference": spec["reference"]})
                if spec["bidirectional"]:
                    edges.append({"source": bus, "target": node, "basis": "modeling_assumption", "reference": spec["reference"]})
    elif construction == "explicit":
        edges = deepcopy(spec.get("edges", []))
        if not isinstance(edges, list):
            raise GraphDataError("Explicit edges must be a list.")
    else:
        raise GraphDataError("construction must be assumed_shared_bus or explicit.")
    seen = set()
    for edge in edges:
        if edge.get("source") not in positions or edge.get("target") not in positions:
            raise GraphDataError("Edge references an unknown node.")
        if edge.get("basis") not in {"modeling_assumption", "documented_physical"} or not isinstance(edge.get("reference"), str) or not edge["reference"].strip():
            raise GraphDataError("Every edge needs an explicit basis and nonempty reference.")
        pair = (edge["source"], edge["target"])
        if pair in seen or pair[0] == pair[1]:
            raise GraphDataError("Duplicate edges or implicit self loops are not supported.")
        seen.add(pair)
    edge_index = np.array([(positions[edge["source"]], positions[edge["target"]]) for edge in edges], dtype=np.int64).reshape(-1, 2).T
    return {"edge_index": edge_index, "node_attributes": np.asarray(attributes, dtype=np.float32),
            "metadata": {"node_order": ids, "nodes": nodes, "edges": edges,
                         "construction": construction, "capacity_scale_kw": scale,
                         "node_attribute_names": [f"is_{kind}" for kind in NODE_TYPES] + ["capacity_scaled", "capacity_known"],
                         "edge_weight_semantics": "unweighted membership; not electrical parameters"}}
