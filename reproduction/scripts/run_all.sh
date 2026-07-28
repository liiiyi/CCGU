#!/usr/bin/env bash
# Full reproduction sweep, in the order the tables are reported.
#
#   bash reproduction/scripts/run_all.sh 2>&1 | tee reproduction/logs/run_all.log
#
# Cells run one at a time (single GPU), so the recorded timings are comparable.
# Reddit is deliberately absent from the sweep.
#
# Two evaluation protocols are reported for every dataset:
#   protect=none  the repository's historical protocol: every original node is
#                 aggregated, and the mapped test set is ~10% of the communities
#                 (about 30 nodes on Cora, so Macro-F1 is noise dominated).
#   protect=test  the original graph's held-out nodes stay in singleton
#                 communities, which is what exp_train's `test_communities`
#                 branch was written for and what makes the numbers comparable in
#                 magnitude to the paper's Table 3.
#
# The post-paper extensions (GAE detector, tail handling, community-scale control)
# are reported as explicit ablations against the same seeds, never as a
# replacement for the baseline rows.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${CCGU_PYTHON:-${REPO_ROOT}/.conda-env/bin/python}"
GRID="${REPO_ROOT}/reproduction/scripts/run_grid.py"
MODELS=(GCN GAT SAGE)
FAILED=0

run () {
  echo
  echo "############ $* ############"
  "${PY}" "${GRID}" "$@" || FAILED=1
}

# A fresh clone has neither directory, and the first write below redirects into
# both, so create them up front.
mkdir -p "${REPO_ROOT}/reproduction/results" "${REPO_ROOT}/reproduction/logs"

echo "############ provenance ############"
"${PY}" "${REPO_ROOT}/reproduction/scripts/report_env.py" \
  --json "${REPO_ROOT}/reproduction/results/environment.json" || FAILED=1
"${PY}" "${REPO_ROOT}/reproduction/scripts/probe_detector_backends.py" \
  --json "${REPO_ROOT}/reproduction/results/detector_backends.json" > \
  "${REPO_ROOT}/reproduction/logs/detector_backends.log" 2>&1 || FAILED=1

# ---- Cora: both protocols, both partition methods, five seeds ---------------
run --datasets cora     --partitions ccp        --models "${MODELS[@]}" --seeds 4 5 6 7 8 --protect test
run --datasets cora     --partitions ccp oslom  --models "${MODELS[@]}" --seeds 4 5        --protect none

# ---- Citeseer --------------------------------------------------------------
run --datasets citeseer --partitions ccp        --models "${MODELS[@]}" --seeds 4 5 6 7 8 --protect test
run --datasets citeseer --partitions ccp oslom  --models "${MODELS[@]}" --seeds 4 5        --protect none

# ---- Coauthor-CS (no shipped split, so the hold-out is derived from the seed)
run --datasets cs       --partitions ccp        --models "${MODELS[@]}" --seeds 4 5        --protect test
run --datasets cs       --partitions ccp        --models "${MODELS[@]}" --seeds 4 5        --protect none

# ---- Pubmed: in the repository's dataset list, not in the paper -------------
run --datasets pubmed   --partitions ccp        --models "${MODELS[@]}" --seeds 4 5        --protect test

# ---- Equation (5) ablation: mapped feature by community mean instead of PCA --
# Both are pre-existing --agg_feat choices; reported side by side rather than
# picking whichever scores better.
run --datasets cora     --partitions ccp        --models "${MODELS[@]}" --seeds 4 5        --protect test --agg_feat mean
run --datasets citeseer --partitions ccp        --models "${MODELS[@]}" --seeds 4 5        --protect test --agg_feat mean

# ---- Post-paper extension: GAE detector vs CCP, same seeds ------------------
run --datasets cora     --partitions gae        --models "${MODELS[@]}" --seeds 4 5        --protect test
run --datasets citeseer --partitions gae        --models "${MODELS[@]}" --seeds 4 5        --protect test

# ---- Post-paper extension: tail handling off (above) vs on -----------------
run --datasets cora     --partitions ccp        --models "${MODELS[@]}" --seeds 4 5        --protect test \
    --extra --ccp_tail_min_size 5
run --datasets cora     --partitions gae        --models "${MODELS[@]}" --seeds 4 5        --protect test \
    --extra --ccp_tail_min_size 5

# ---- Post-paper extension: community-scale control ------------------------
# Coarser (theta 40) and finer (theta 10) than the default 20, plus a size cap.
run --datasets cora     --partitions ccp        --models "${MODELS[@]}" --seeds 4 5        --protect test \
    --extra --ccp_theta 10
run --datasets cora     --partitions ccp        --models "${MODELS[@]}" --seeds 4 5        --protect test \
    --extra --ccp_theta 40
run --datasets cora     --partitions ccp        --models "${MODELS[@]}" --seeds 4 5        --protect test \
    --extra --ccp_max_community_size 40

echo
echo "############ tables ############"
"${PY}" "${REPO_ROOT}/reproduction/scripts/make_tables.py" \
  --out "${REPO_ROOT}/reproduction/results/tables.md" > /dev/null || FAILED=1
echo "wrote reproduction/results/tables.md"

if [ "${FAILED}" -ne 0 ]; then
  echo "SWEEP INCOMPLETE: at least one grid reported a failure" >&2
  exit 1
fi
echo "SWEEP COMPLETE"
