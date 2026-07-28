# Community detection and the mapping stage

This guide documents the partitioning options in `exp/methods/` and the flags that
control them. See the [root README](../../README.md) for the method summary and the
pipeline commands.

The partition stage determines the mapped graph, and therefore everything computed
from it. It runs once per deployment and its result is reused by every subsequent
deletion request, so a deterministic partition is preferable to a stochastic one.

---

## Available methods

`--partition` selects the detector. All previously available options are retained
with their original behaviour and defaults; the default remains `oslom`.

| Option | Implementation |
|---|---|
| `ccp` | `CCP.py` — coarse Louvain initialisation followed by fine-grained refinement, with overlapping membership |
| `gae` | `GAEPartition.py` — graph-auto-encoder embedding followed by clustering |
| `oslom`, `slpa`, `lpa` | label-propagation variants (`OSLOM.py`, `SLPA.py`, `LPA.py`) |
| `louvain` | `Louvain.py`; uses `cdlib` when installed, otherwise `networkx`. Resolution is set by `--louvain_resolution` (default `2.0`), kept separate from `--ccp_resolution` |
| `infomap` | `Infomap.py`; requires the `infomap` package |
| `nikm`, `test` | `NIKM.py`, `CCD.py`; require `hnswlib`, which is not in the pinned dependency set |

## CCP

`CCP.py` implements the coarse-to-fine overlapping community process described in
the paper, in a form that is deterministic given `--random_seed` within a fixed
software environment:

1. **Coarse level** — seeded Louvain modularity maximisation on the simple
   undirected projection of the graph.
2. **Fine level** — communities larger than `--ccp_theta` are re-partitioned by
   Louvain on their induced subgraph, recursively, until each is at most `theta`
   nodes; blocks that do not split are divided deterministically in breadth-first
   order. Setting `--ccp_conductance_threshold` to a non-negative value restricts
   refinement to communities whose conductance exceeds it, which follows the
   paper's variant for large graphs.
3. **Overlapping membership** — a node may additionally join up to
   `--ccp_max_communities_per_node - 1` neighbouring communities whose *belonging
   coefficient* clears `--ccp_overlap_threshold`. Belonging is the geometric mean of
   a structural term, the fraction of a node's neighbours inside the candidate
   community, and an attribute term, the cosine similarity between the node and the
   community centroid in a graph-smoothed feature space obtained by
   `--ccp_propagation_steps` propagation steps. Because the same smoothed signal
   underlies the mapped features and labels, the attribute term is consistent with
   the mapping that follows.

Stage order is fixed: coarse, fine, overlap, singleton backfill, size cap,
small-community merge.

`--ccp_theta` is a *target*, not a guarantee. Neither `--ccp_theta` nor
`--ccp_max_community_size` fixes the final maximum community size: conductance
filtering can leave an oversized community unrefined, and the small-community
consolidation that runs after the size cap can merge blocks back above it. Read the
realised size distribution from the diagnostics rather than assuming either value.

## Optional GAE detector

`GAEPartition.py` offers `--partition gae` as an alternative. A two-layer DGL
`GraphConv` encoder is trained on a link-reconstruction objective — observed edges
against sampled non-edges, drawn by deterministic rejection sampling against the
adjacency — and the learned representation is then clustered, with the number of
clusters derived from `--ccp_theta`. No labels and no evaluation metric enter the
objective. The overlap assignment, scale controls, protected singletons, diagnostics
and identifier compaction are shared with CCP, so the returned mappings have the
same structure.

Relevant flags: `--gae_hidden_dim` (`64`), `--gae_latent_dim` (`16`), `--gae_epochs`
(`100`), `--gae_learning_rate` (`0.01`), `--gae_device` (`auto`), and
`--gae_round_decimals` (`6`), which rounds the normalised embedding before clustering
so that small floating-point differences do not move a cluster boundary.

## Scale controls

The number of mapped nodes governs both the size of the mapped graph and the cost of
updating it after a deletion.

| Flag | Default | Effect |
|---|---|---|
| `--ccp_theta` | `20` | target community size; oversized communities are refined |
| `--ccp_resolution`, `--ccp_fine_resolution` | `1.0` | Louvain resolutions; higher values usually give more, smaller communities |
| `--ccp_max_community_size` | `0` (disabled) | upper bound applied after the overlap stage |
| `--ccp_max_communities_per_node` | `3` | maximum memberships per node |
| `--ccp_overlap_threshold` | `0.35` | minimum belonging coefficient |

### Starting ranges

Starting points for exploration, not tuned optima.

| Setting | Small graph | Medium graph | Large graph / cost-first |
|---|---|---|---|
| `--ccp_theta` | `20` | `40`–`60` | `60`–`100` |
| `--ccp_resolution`, `--ccp_fine_resolution` | `1.0` | `1.0` | `1.0` |
| `--ccp_tail_min_size` | `3`–`5` | `3`–`5` | `3`–`5` |
| `--ccp_max_community_size` | about `2`–`4` × `theta` | about `2`–`4` × `theta` | about `2`–`4` × `theta` |

Raise the resolutions above `1.0` only if the diagnostics show that communities
remain too coarse after refinement. Adjust `--ccp_overlap_threshold` before
`--ccp_max_communities_per_node`: more overlap yields a richer mapped graph but makes
each deletion touch more communities. A smaller `theta` gives more mapped nodes,
more training signal and finer mapped labels; a larger `theta` gives a smaller mapped
graph and a cheaper update, at the cost of coarser mapped features and labels.

## Small communities and singletons

Singleton and long-tail communities become sparse mapped nodes: they carry little
signal and weaken the mapped feature and label rules, and in quantity they can
destabilise the partition and the training that follows.

`--ccp_tail_min_size` (default `0`, disabled) merges a community below the given size
into the neighbouring community with the highest attachment score, computed from the
observed adjacency and the feature centroids only, so no labels enter the mapping.
Small communities are never merge targets, so merges do not chain; a small community
with no eligible neighbour is left unchanged, and isolated nodes are kept.

## Evaluation-node protection

`--ccp_protect_eval_nodes` (`none`, `test`, `test_val`; default `none`) keeps the
original graph's held-out nodes in singleton communities rather than aggregating
them, so held-out nodes are not absorbed into a training community. The training
stage treats such singleton communities as mapped evaluation nodes.

## Diagnostics and fair selection

`partition_diagnostics.py` summarises every partition: community count, compression
ratio, coverage, the histogram of memberships per node, the community-size
distribution, counts of singleton and very small communities, label purity, the
modularity of the disjoint projection, and wall-clock cost. The summary is logged and
written next to the community file.

`--partition_min_communities` (default `8`) rejects a partition too small to support
a train/validation/test split, or one that leaves original nodes unrepresented.
`--partition_max_retries` (default `0`) may re-run a rejected partition with the next
seed. The acceptance criterion is a function of the partition alone, never of model
quality, and every attempt is recorded.

Select parameters and seeds using these structural diagnostics or a validation split
only, never test metrics, and never by keeping whichever seed scored best. Label
purity is the one diagnostic computed from labels, and the field reported here is
computed over the full label set: it is **descriptive only and must not be used for
parameter or seed selection**. If a label-based quantity is needed for selection,
compute it strictly from training or validation labels.

## Seeds and cached artifacts

`--random_seed` seeds NumPy, Python `random`, torch, community detection, the
train/validation/test split and the sampled deletion request. Cached artifact names
do not encode the seed, so `config.py` accepts `CCGU_DATA_ROOT` and `CCGU_DGL_DATA`;
setting the former per configuration keeps artifacts separate while the latter allows
one shared dataset cache. `reproduction/scripts/run_experiment.py` sets both. Reuse
of a cached community file is reported in the log, because the seed on that command
line did not influence it.

Community detection is stochastic. Vary `--random_seed`, report a spread rather than
a single draw, and read the diagnostics before the model metrics.

## Execution scope

Only the graph-auto-encoder encoder runs on the GPU (`--gae_device auto|cuda|cpu`,
falling back to CPU when CUDA is unavailable). Its clustering step, the overlap
assignment and the diagnostics run on the CPU.

The classical methods — Louvain, Infomap and the label-propagation variants — are
Python or CPU-bound and have no dedicated GPU-accelerated implementation. Some
legacy wrappers move DGL tensors onto the selected device, but this does not remove
the CPU-bound loop around them and should not be read as GPU acceleration.
`reproduction/scripts/probe_detector_backends.py` reports which optional backends are
importable and where each detector executes.
