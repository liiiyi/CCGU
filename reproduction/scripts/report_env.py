#!/usr/bin/env python
"""Print (and optionally persist) the exact software/hardware provenance of a run.

    .conda-env/bin/python reproduction/scripts/report_env.py
    .conda-env/bin/python reproduction/scripts/report_env.py --json reproduction/results/environment.json

Also serves as the post-install smoke check for reproduction/scripts/setup_env.sh:
it exercises a real DGL GPU message-passing op for each of the three backbones
used by CCGU, so a broken CUDA install fails here rather than mid-experiment.
"""

import argparse
import json
import os
import platform
import subprocess
import sys


def _run(command):
    try:
        return subprocess.check_output(
            command, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:  # pragma: no cover - provenance is best effort
        return None


def collect():
    import dgl
    import numpy
    import scipy
    import sklearn
    import networkx
    import torch

    info = {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "dgl": dgl.__version__,
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__,
            "networkx": networkx.__version__,
        },
        "cuda": {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "devices": [],
            "driver": _run(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader",
                ]
            ),
        },
        "cpu_count": _run(["nproc"]),
    }
    for index in range(torch.cuda.device_count()):
        info["cuda"]["devices"].append(
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": ".".join(
                    str(part) for part in torch.cuda.get_device_capability(index)
                ),
                "total_memory_gb": round(
                    torch.cuda.get_device_properties(index).total_memory / 1024 ** 3, 1
                ),
            }
        )
    return info


def gpu_backbone_check():
    """Run one forward pass of each backbone through DGL on the GPU (or CPU)."""
    import dgl
    import torch
    from dgl.nn.pytorch import GATConv, GraphConv, SAGEConv

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    graph = dgl.graph(([0, 1, 2], [1, 2, 0])).to(device)
    features = torch.randn(3, 8, device=device)
    shapes = {
        "GCN": tuple(GraphConv(8, 4).to(device)(graph, features).shape),
        "SAGE": tuple(SAGEConv(8, 4, "mean").to(device)(graph, features).shape),
        "GAT": tuple(GATConv(8, 4, 2).to(device)(graph, features).shape),
    }
    return {"device": device, "output_shapes": shapes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", help="also write the report to this path")
    args = parser.parse_args()

    info = collect()
    info["backbone_forward_check"] = gpu_backbone_check()

    for key in ("python", "platform", "cpu_count"):
        print("{:<16}{}".format(key, info[key]))
    for name, version in info["packages"].items():
        print("{:<16}{}".format(name, version))
    print("{:<16}{}".format("cuda_available", info["cuda"]["available"]))
    print("{:<16}{}".format("cuda_driver", info["cuda"]["driver"]))
    for device in info["cuda"]["devices"]:
        print(
            "{:<16}{} (sm_{}, {} GB)".format(
                "cuda:%d" % device["index"],
                device["name"],
                device["capability"].replace(".", ""),
                device["total_memory_gb"],
            )
        )
    print(
        "{:<16}{} {}".format(
            "backbone check",
            info["backbone_forward_check"]["device"],
            info["backbone_forward_check"]["output_shapes"],
        )
    )

    if args.json:
        directory = os.path.dirname(os.path.abspath(args.json))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(args.json, "w") as handle:
            json.dump(info, handle, indent=2, sort_keys=True)
        print("{:<16}{}".format("wrote", args.json))


if __name__ == "__main__":
    main()
