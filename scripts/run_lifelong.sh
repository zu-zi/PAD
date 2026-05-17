#!/usr/bin/env bash
# Train PAD sequentially over the five seen domains declared in
# configs/pad.yml, evaluating on every seen domain after each step
# and appending to Results/seq_eval.csv.
#
# Usage:
#   bash scripts/run_lifelong.sh                       # start from the first domain
#   bash scripts/run_lifelong.sh <START_IDX|DOMAIN>    # resume partway
#   bash scripts/run_lifelong.sh <START> <RESUME_CKPT> # resume with explicit ckpt

set -euo pipefail

CFG="configs/pad.yml"
MODEL_NAME="ViT-B-16"
CKPT="${MODEL_NAME}_stage2.pth"
ROOT_DIR="Results"
CSV_OUT="${ROOT_DIR}/seq_eval.csv"
mkdir -p "${ROOT_DIR}"

# Read the domain list from the YAML (order preserved).
mapfile -t DOMAINS < <(python - "${CFG}" <<'PY'
import sys, yaml
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)
for d in cfg.get("DOMAINS", []):
    print(d["name"])
PY
)
if [[ ${#DOMAINS[@]} -eq 0 ]]; then
  echo "[ERR] no DOMAINS declared in ${CFG}"; exit 1
fi

START_ARG="${1:-0}"
EXPLICIT_RESUME="${2:-}"

start_idx=-1
if [[ "${START_ARG}" =~ ^[0-9]+$ ]]; then
  start_idx="${START_ARG}"
else
  for i in "${!DOMAINS[@]}"; do
    [[ "${DOMAINS[$i]}" == "${START_ARG}" ]] && start_idx="${i}" && break
  done
fi
if (( start_idx < 0 || start_idx >= ${#DOMAINS[@]} )); then
  echo "[ERR] invalid start: '${START_ARG}'"; exit 1
fi

prev_ckpt=""
if [[ -n "${EXPLICIT_RESUME}" ]]; then
  [[ -f "${EXPLICIT_RESUME}" ]] || { echo "[ERR] ckpt not found: ${EXPLICIT_RESUME}"; exit 1; }
  prev_ckpt="${EXPLICIT_RESUME}"
elif (( start_idx > 0 )); then
  prev="${DOMAINS[$((start_idx-1))]}"
  candidate="${ROOT_DIR}/${prev}/${CKPT}"
  [[ -f "${candidate}" ]] && prev_ckpt="${candidate}" \
    || echo "[WARN] previous checkpoint not found: ${candidate}"
fi

last=$(( ${#DOMAINS[@]} - 1 ))
for (( idx="${start_idx}"; idx<=last; idx++ )); do
  domain="${DOMAINS[$idx]}"
  domain_out="${ROOT_DIR}/${domain}"
  mkdir -p "${domain_out}"

  echo "[Stage] idx=${idx}  domain=${domain}  resume=${prev_ckpt:-<none>}"

  if [[ -n "${prev_ckpt}" ]]; then
    python train_lifelong.py --config_file "${CFG}" --domain_idx "${idx}" \
      --resume_ckpt "${prev_ckpt}" OUTPUT_DIR "${domain_out}"
  else
    python train_lifelong.py --config_file "${CFG}" --domain_idx "${idx}" \
      OUTPUT_DIR "${domain_out}"
  fi

  cur_ckpt="${domain_out}/${CKPT}"
  [[ -f "${cur_ckpt}" ]] || { echo "[ERR] checkpoint missing: ${cur_ckpt}"; exit 1; }

  seen=()
  for j in $(seq 0 ${idx}); do seen+=("${DOMAINS[$j]}"); done
  eval_csv=$(IFS=, ; echo "${seen[*]}")

  python test_lifelong.py --config_file "${CFG}" --domain_idx "${idx}" \
    --ckpt "${cur_ckpt}" --trained_domain "${domain}" \
    --eval_domains "${eval_csv}" --csv_out "${CSV_OUT}" --outdir "${domain_out}"

  prev_ckpt="${cur_ckpt}"
done

echo "[DONE] results at ${CSV_OUT}"
