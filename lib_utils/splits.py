"""Shared definition of the original-graph held-out split.

Both the mapping stage (when it is asked to keep evaluation nodes out of
communities) and the training stage (when it decides which singleton communities
count as mapped test nodes) need to agree, byte for byte, on which original nodes
are held out.  Deriving it in two places invites a silent disagreement, so it
lives here.
"""

import numpy as np
from sklearn.model_selection import train_test_split


def original_holdout_mask(graph, num_nodes, test_ratio, seed, include_val=False):
    """Boolean mask over ORIGINAL node ids marking the held-out evaluation nodes.

    Uses the dataset's own ``test_mask`` (and optionally ``val_mask``) when DGL
    provides one -- Cora/Citeseer/Pubmed ship the standard Planetoid split -- and
    otherwise derives a deterministic split from ``(test_ratio, seed)``, which is
    what Coauthor-CS needs.
    """
    node_data = getattr(graph, "ndata", {}) or {}
    if "test_mask" in node_data:
        mask = node_data["test_mask"].cpu().numpy().astype(bool)
        if include_val and "val_mask" in node_data:
            mask = mask | node_data["val_mask"].cpu().numpy().astype(bool)
        return mask

    indices = np.arange(num_nodes)
    _, held_out = train_test_split(
        indices, test_size=test_ratio, random_state=seed
    )
    mask = np.zeros(num_nodes, dtype=bool)
    mask[held_out] = True
    return mask
