#!/usr/bin/env python
"""Report which community-detection backends are actually available here.

    .conda-env/bin/python reproduction/scripts/probe_detector_backends.py \
        --json reproduction/results/detector_backends.json

The point is to keep the README honest about the difference between "GPU
accelerated" and "we happen to move a tensor to CUDA".  A legacy detector counts
as GPU accelerated only if an installed backend runs the *whole* algorithm on the
device; that is checked here rather than assumed.
"""

import argparse
import importlib.util
import json
import os
import platform
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

#: name -> (what it would accelerate, why we care)
OPTIONAL_BACKENDS = {
    "cugraph": "GPU Louvain / Leiden / ECG (RAPIDS)",
    "pylibcugraph": "RAPIDS C++ graph primitives",
    "nx_cugraph": "networkx dispatch to cuGraph",
    "cudf": "RAPIDS dataframes, required by cugraph",
    "metis": "METIS k-way partitioning (CPU)",
    "pymetis": "METIS bindings (CPU)",
    "leidenalg": "Leiden via igraph (CPU)",
    "igraph": "igraph algorithms (CPU)",
    "infomap": "Infomap map equation (CPU)",
    "hnswlib": "approximate nearest neighbours, for --partition test / nikm",
    "cdlib": "community-detection library wrappers (CPU)",
}

#: The detectors the CLI exposes, and where each one actually runs.
DETECTOR_EXECUTION = [
    ("ccp", "CPU", "networkx Louvain + numpy/scipy; no GPU backend used"),
    ("gae", "GPU encoder + CPU clustering",
     "DGL GraphConv encoder trains on CUDA when available; k-means, overlap and "
     "tail repair are CPU"),
    ("oslom", "CPU (legacy)",
     "label propagation in Python; moves the graph to CUDA only to call "
     "successors() per node, which makes it slower, not faster"),
    ("slpa", "CPU (legacy)", "label propagation in Python"),
    ("infomap", "CPU", "infomap C++ extension, CPU only"),
    ("louvain", "CPU", "cdlib if installed, else networkx louvain_communities"),
    ("lpa", "CPU (legacy)", "label propagation in Python"),
    ("nikm", "CPU clustering on GPU embeddings", "k-means over GNN embeddings"),
    ("test", "CPU", "CCD via hnswlib"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", help="also write the report to this path")
    options = parser.parse_args()

    report = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "backends": {},
        "detector_execution": [
            {"detector": name, "runs_on": where, "note": note}
            for name, where, note in DETECTOR_EXECUTION
        ],
    }

    try:
        import torch

        report["cuda_available"] = bool(torch.cuda.is_available())
        report["torch"] = torch.__version__
    except ImportError:
        report["cuda_available"] = False
        report["torch"] = None

    print("python           {}".format(report["python"]))
    print("torch            {}".format(report["torch"]))
    print("cuda available   {}".format(report["cuda_available"]))
    print()
    print("optional community-detection backends:")
    for name, purpose in sorted(OPTIONAL_BACKENDS.items()):
        available = importlib.util.find_spec(name) is not None
        version = None
        if available:
            try:
                version = getattr(__import__(name), "__version__", None)
            except Exception:  # pragma: no cover - import side effects
                version = None
        report["backends"][name] = {
            "available": available,
            "version": version,
            "purpose": purpose,
        }
        print("  {:<14} {:<14} {}".format(
            name,
            ("yes " + version) if (available and version) else ("yes" if available else "NO"),
            purpose,
        ))

    gpu_graph_backend = any(
        report["backends"][name]["available"]
        for name in ("cugraph", "pylibcugraph", "nx_cugraph")
    )
    report["gpu_graph_backend_available"] = gpu_graph_backend
    print()
    print("GPU graph backend for Louvain/Leiden/METIS/Infomap: {}".format(
        "AVAILABLE" if gpu_graph_backend else "NOT AVAILABLE"))
    if not gpu_graph_backend:
        print("  No compatible RAPIDS/cuGraph package is importable in this Python {}".format(
            report["python"]))
        print("  environment.  The legacy detectors therefore run on the CPU, and this")
        print("  reproduction makes no GPU-acceleration claim for them.")
    print()
    print("where each --partition choice actually runs:")
    for entry in report["detector_execution"]:
        print("  {:<9} {:<32} {}".format(
            entry["detector"], entry["runs_on"], entry["note"]))

    if options.json:
        directory = os.path.dirname(os.path.abspath(options.json))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(options.json, "w") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        print("\nwrote {}".format(options.json))


if __name__ == "__main__":
    main()
