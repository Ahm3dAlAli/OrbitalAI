#!/bin/bash
# OrbitSight — regenerate ALL submission assets in one shot.
#
# Runs every generator and (optionally) the evaluator, producing:
#   docs/figures/*.png        banner, architecture, result charts, sample sheet
#   docs/latency_cpu.md/json  measured latency (if a model is present)
#   docs/vis/*.gif|png        detection demo + failure gallery (if data present)
#   docs/OrbitSight_Proposal.pdf   the 5-page proposal (if Chrome present)
#   Evaluation_Metrics.xlsx   fresh scorecard for a prediction dir (if data present)
#
# Data-free steps always run (they render from results.json). Data-dependent
# steps are guarded and skipped with a message if inputs are missing.
#
#   bash scripts/build_all.sh                       # everything auto-detected
#   PRED_DIR=predictions/router_ctta bash scripts/build_all.sh   # score a specific dir
set -u
cd "$(dirname "$0")/.." || exit 1
export KMP_DUPLICATE_LIB_OK=TRUE PYTHONUNBUFFERED=1

DATA="${DATA_DIR:-OrbitSight_Dataset/Testing_sets}"
PRED="${PRED_DIR:-predictions/testing_router2}"
MODEL="${MODEL:-}"                       # auto-picked below if empty
DEMO_SEQ="${DEMO_SEQ:-DAVIS_SAOCOM1B_46265_2024-12-04-18-21-37}"
PY=python3
ok(){ echo "  ✓ $*"; }; skip(){ echo "  – skip: $*"; }
hr(){ echo "── $* ──────────────────────────────────────────"; }

hr "1/6  static figures (from results.json — no data needed)"
$PY scripts/make_banner.py   && ok banner
$PY scripts/make_arch.py     && ok "architecture diagram"
$PY scripts/make_figures.py  && ok "result charts"

hr "2/6  sample detections (needs data + predictions)"
if [ -d "$DATA" ] && [ -d "$PRED" ]; then
  $PY scripts/make_samples.py --data-dir "$DATA" --pred-dir "$PRED" \
      --gt-dir "$DATA" --out docs/figures/sample_detections.png && ok "sample sheet"
else
  skip "no $DATA or $PRED"
fi

hr "3/6  latency benchmark (needs a model + data)"
if [ -z "$MODEL" ]; then
  for m in models/g192_ctx.pt models/evt_centernet_aug.pt models/evt_centernet.pt; do
    [ -f "$m" ] && MODEL="$m" && break
  done
fi
if [ -n "$MODEL" ] && [ -d "$DATA" ]; then
  $PY scripts/benchmark_latency.py --device "${DEVICE:-cpu}" --model "$MODEL" \
      --data-dir "$DATA" --batch 1 --md-out docs/latency_cpu.md \
      --json-out docs/latency.json && ok "latency ($MODEL)"
else
  skip "no model or data for latency"
fi

hr "4/6  visualization demo + failure gallery (needs data + predictions)"
if [ -d "$DATA" ] && [ -d "$PRED" ] && [ -f "$DATA/${DEMO_SEQ}_labeled_events.npy" ]; then
  $PY scripts/visualize.py --data-dir "$DATA" --seq "$DEMO_SEQ" \
      --pred-dir "$PRED" --gt-dir "$DATA" --max-frames 60 --fps 8 \
      --out "docs/vis/${DEMO_SEQ}_demo.gif" && ok "demo gif"
  $PY scripts/visualize.py --data-dir "$DATA" --seq "$DEMO_SEQ" \
      --pred-dir "$PRED" --gt-dir "$DATA" --gallery --gallery-n 8 \
      --out "docs/vis/${DEMO_SEQ}_failures.png" && ok "failure gallery"
else
  skip "no data/predictions for $DEMO_SEQ"
fi

hr "5/6  evaluate predictions -> Evaluation_Metrics.xlsx (needs GT)"
if [ -d "$DATA" ] && [ -d "$PRED" ]; then
  $PY Dataloader/evaluate.py --gt-dir "$DATA" --pred-dir "$PRED" \
      --excel-out Evaluation_Metrics.xlsx 2>/dev/null \
    && ok "scored $PRED -> Evaluation_Metrics.xlsx" \
    || $PY scripts/evaluate_wrapper.py --dataset "OrbitSight_Dataset" \
         --pred-dir "$PRED" --excel-out Evaluation_Metrics.xlsx && ok "scored $PRED"
else
  skip "no data/predictions to score"
fi

hr "6/6  render the 5-page proposal PDF (needs Chrome)"
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
[ -x "$CHROME" ] || CHROME="$(command -v google-chrome || command -v chromium || true)"
if [ -n "$CHROME" ] && [ -f docs/OrbitSight_Proposal.html ]; then
  "$CHROME" --headless --disable-gpu --no-pdf-header-footer \
    --print-to-pdf="docs/OrbitSight_Proposal.pdf" \
    "file://$PWD/docs/OrbitSight_Proposal.html" 2>/dev/null && ok "proposal PDF"
else
  skip "no Chrome or proposal HTML"
fi

echo
hr "DONE — key outputs"
ls -1 docs/figures/*.png 2>/dev/null | sed 's/^/  /'
[ -f docs/OrbitSight_Proposal.pdf ] && echo "  docs/OrbitSight_Proposal.pdf"
[ -f Evaluation_Metrics.xlsx ] && echo "  Evaluation_Metrics.xlsx"
[ -f results.json ] && echo "  results.json  (canonical measured results)"
