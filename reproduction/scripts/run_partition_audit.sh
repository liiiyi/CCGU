#!/usr/bin/env bash
# Community-detection audit: quality and cost of every wired partition method,
# measured with the partition stage in isolation and a fixed thread count.
#
#   bash reproduction/scripts/run_partition_audit.sh
#
# Writes one JSON per (dataset, method, seed) under
# reproduction/results/partition/ and a markdown table at
# reproduction/results/partition_audit.md.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${CCGU_PYTHON:-${REPO_ROOT}/.conda-env/bin/python}"
AUDIT="${REPO_ROOT}/reproduction/scripts/audit_partition.py"
OUT="${REPO_ROOT}/reproduction/results/partition"
THREADS="${CCGU_AUDIT_THREADS:-8}"
FAILED=0

# A fresh clone has neither directory; the loops below write into both.
mkdir -p "${OUT}" "${REPO_ROOT}/reproduction/logs"

# Legacy oslom/slpa consume no randomness at all, so one seed characterises them
# completely; that is itself an audit finding and is asserted below.
for dataset in cora citeseer; do
  for method in ccp oslom slpa infomap; do
    for seed in 4 5; do
      if [ "${method}" != "ccp" ] && [ "${seed}" != "4" ]; then
        # still run seed 5 for the legacy methods once, to demonstrate that the
        # result does not depend on the seed
        if [ "${dataset}" != "cora" ]; then continue; fi
      fi
      echo "### ${dataset} / ${method} / seed ${seed}"
      "${PY}" "${AUDIT}" --dataset "${dataset}" --method "${method}" --seed "${seed}" \
        --num-threads "${THREADS}" \
        --json "${OUT}/${dataset}_${method}_seed${seed}.json" \
        > "${REPO_ROOT}/reproduction/logs/audit_${dataset}_${method}_seed${seed}.log" 2>&1 \
        || { echo "FAILED ${dataset}/${method}/seed${seed}"; FAILED=1; }
    done
  done
done

# ccp only on the larger graphs: the legacy label-propagation cost grows with the
# square of the iteration count, which is exactly the scaling problem documented
# in README.md.
for dataset in cs pubmed; do
  for seed in 4 5; do
    echo "### ${dataset} / ccp / seed ${seed}"
    "${PY}" "${AUDIT}" --dataset "${dataset}" --method ccp --seed "${seed}" \
      --num-threads "${THREADS}" \
      --json "${OUT}/${dataset}_ccp_seed${seed}.json" \
      > "${REPO_ROOT}/reproduction/logs/audit_${dataset}_ccp_seed${seed}.log" 2>&1 \
      || { echo "FAILED ${dataset}/ccp/seed${seed}"; FAILED=1; }
  done
done

"${PY}" "${REPO_ROOT}/reproduction/scripts/make_partition_table.py" \
  --out "${REPO_ROOT}/reproduction/results/partition_audit.md" || FAILED=1

if [ "${FAILED}" -ne 0 ]; then
  echo "AUDIT INCOMPLETE" >&2
  exit 1
fi
echo "AUDIT COMPLETE"
