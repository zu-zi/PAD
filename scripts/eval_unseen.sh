#!/usr/bin/env bash
# Evaluate a PAD checkpoint on the 7 unseen datasets using the
# standard protocol per dataset (matches Table 1's Unseen-Avg column):
# GRID / PRID / iLIDS average over 10 random splits; VIPeR / CUHK01
# average over 20 runs (10 identity-splits x 2 directions);
# CUHK02 / SenseReID use a single deterministic split.
#
# Usage:
#   bash scripts/eval_unseen.sh <CKPT_PATH> [OUTDIR]

set -euo pipefail

CKPT="${1:-}"
[[ -n "${CKPT}" ]] || { echo "Usage: $0 <CKPT_PATH> [OUTDIR]"; exit 1; }
[[ -f "${CKPT}" ]] || { echo "[ERR] ckpt not found: ${CKPT}"; exit 1; }

OUTDIR="${2:-Results_unseen/$(basename "${CKPT%.*}")}"
mkdir -p "${OUTDIR}"
CSV_RAW="${OUTDIR}/unseen_raw.csv"
CSV_OUT="${OUTDIR}/unseen_eval.csv"
rm -f "${CSV_RAW}" "${CSV_OUT}"

run_eval () {
  local domain="$1" split="$2" direction="$3"
  python test_lifelong.py --ckpt "${CKPT}" \
    --eval_domains "${domain}" --csv_out "${CSV_RAW}" --outdir "${OUTDIR}" \
    EVAL.SPLIT_ID "${split}" EVAL.DIRECTION "${direction}" EVAL.SEED 123
}

for d in cuhk02 sensereid; do
  echo "[unseen] ${d}"
  run_eval "${d}" 0 auto
done
for d in grid ilids prid; do
  echo "[unseen] ${d} (10 splits)"
  for sid in $(seq 0 9); do run_eval "${d}" "${sid}" auto; done
done
for d in viper cuhk01; do
  echo "[unseen] ${d} (20 splits)"
  for sid in $(seq 0 19); do run_eval "${d}" "${sid}" auto; done
done

python - "${CSV_RAW}" "${CSV_OUT}" "${CKPT}" <<'PY'
import csv
import os
import statistics
import sys

raw, out, ckpt = sys.argv[1], sys.argv[2], os.path.basename(sys.argv[3])
domains = ["cuhk01", "cuhk02", "grid", "ilids", "prid", "sensereid", "viper"]
vals = {d: {"r1": [], "m": []} for d in domains}

with open(raw) as f:
    for row in csv.DictReader(f):
        d = row["eval_domain"]
        if d in vals:
            vals[d]["r1"].append(float(row["rank1"]))
            vals[d]["m"].append(float(row["map"]))

rows = [(d, statistics.mean(v["r1"]), statistics.mean(v["m"]))
        for d, v in vals.items() if v["r1"]]
if not rows:
    raise SystemExit("[ERR] no rows collected")

avg_r1 = sum(r for _, r, _ in rows) / len(rows)
avg_m = sum(m for _, _, m in rows) / len(rows)

with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["eval_domain", "Rank-1", "mAP", "ckpt"])
    for d, r1, m in rows:
        w.writerow([d, f"{r1*100:.2f}", f"{m*100:.2f}", ckpt])
    w.writerow(["Unseen-Avg", f"{avg_r1*100:.2f}", f"{avg_m*100:.2f}", ckpt])

print(f"Unseen-Avg: Rank-1={avg_r1*100:.2f}%  mAP={avg_m*100:.2f}%")
PY

rm -f "${CSV_RAW}"
echo "[OK] wrote ${CSV_OUT}"
