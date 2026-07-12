# OrbitSight — train / infer / evaluate driver.
# Usage:  make train | make infer | make eval | make all | make viz
# Override paths:  make infer DATA=OrbitSight_Dataset MODEL=models/coherence_lgbm.joblib

export KMP_DUPLICATE_LIB_OK = TRUE        # guard duplicate libomp on conda/macOS

DATA   ?= OrbitSight_Dataset
MODEL  ?= models/coherence_lgbm.joblib
PRED   ?= predictions
PY     ?= python3

.PHONY: all train infer infer-train infer-test eval viz clean help

help:
	@echo "make train       - train the coherence classifier on Training_sets"
	@echo "make infer       - run inference on Training_sets and Testing_sets"
	@echo "make eval        - score predictions -> Evaluation_Metrics.xlsx"
	@echo "make all         - train + infer + eval"
	@echo "make viz SEQ=...  - render overlays + (x,y,t) plot for one sequence"

all: train infer eval

train:
	$(PY) scripts/train.py --data-dir $(DATA)/Training_sets --out $(MODEL)

infer: infer-test infer-train

infer-test:
	$(PY) scripts/infer.py --data-dir $(DATA)/Testing_sets  --model $(MODEL) --out-dir $(PRED)/testing

infer-train:
	$(PY) scripts/infer.py --data-dir $(DATA)/Training_sets --model $(MODEL) --out-dir $(PRED)/training

eval:
	$(PY) Dataloader/evaluate.py \
	  --train-gt-dir $(DATA)/Training_sets --train-pred-dir $(PRED)/training \
	  --test-gt-dir  $(DATA)/Testing_sets  --test-pred-dir  $(PRED)/testing \
	  --excel-out Evaluation_Metrics.xlsx

# Evaluate only the test split (faster):
eval-test:
	$(PY) Dataloader/evaluate.py --gt-dir $(DATA)/Testing_sets \
	  --pred-dir $(PRED)/testing --excel-out Evaluation_Metrics.xlsx

SEQ ?= DAVIS_SL12RB2_15772_2024-12-04-18-21-37
viz:
	$(PY) scripts/visualize.py --sequences $(SEQ) \
	  --data-dir $(DATA)/Training_sets --model $(MODEL) --out-dir output

clean:
	rm -rf $(PRED)/training/* $(PRED)/testing/* output/* Evaluation_Metrics.xlsx
