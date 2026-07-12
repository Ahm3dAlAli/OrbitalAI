# Running OrbitSight on **rolf** (GPU)

The whole deep pipeline is GPU-ready. On an RTX 2080 Ti an epoch is **seconds**
(vs ~800 s for grid-192 on a laptop CPU), so the 16-epoch CPU cap is gone — we
train with `--epochs 40 --patience 6` (early stopping) instead.

> **Why 40 + early stop, not 16?** 16 was purely a CPU wall-clock budget. With a
> validation split + early stopping, you set a high cap and the run stops itself
> when validation loss stops improving — no wasted epochs, no guessing.

---

## 0. One-time: copy the project + data to rolf

From your laptop (this repo lives at `OrbitalAI/`):

```bash
# code (small) -> home dir (backed up)
rsync -av --exclude OrbitSight_Dataset --exclude '*.pt' --exclude predictions \
    OrbitalAI/  <user>@rolf.ifi.uzh.ch:~/OrbitalAI/

# dataset (~6 GB) -> fast local scratch (NOT home; home is quota-limited)
ssh <user>@rolf.ifi.uzh.ch 'mkdir -p /local/scratch/<user>/OrbitSight_Dataset && chgrp aiml /local/scratch/<user>'
rsync -av OrbitalAI/OrbitSight_Dataset/  <user>@rolf.ifi.uzh.ch:/local/scratch/<user>/OrbitSight_Dataset/
```

Then symlink the data into the project on rolf:

```bash
ssh <user>@rolf.ifi.uzh.ch
cd ~/OrbitalAI
ln -s /local/scratch/<user>/OrbitSight_Dataset OrbitSight_Dataset
```

## 1. Environment (miniconda, per the group guide)

```bash
# install miniconda to local scratch (keeps home quota free), then:
conda env create -f environment.yml        # creates 'orbitsight'
conda activate orbitsight
# if the pinned pytorch-cuda fails, use the group's line:
#   conda install pytorch pytorch-cuda=11.8 torchvision -c pytorch -c nvidia
python3 -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 2. Pick a free GPU and run in a screen (survives disconnects)

```bash
nvidia-smi                     # find an idle GPU (run sparingly)
screen -S orbit
conda activate orbitsight
cd ~/OrbitalAI
# be nice (low priority) and pin ONE gpu:
CUDA_VISIBLE_DEVICES=0 nice -n 15 bash scripts/run_rolf.sh 2>&1 | tee run_rolf.log
# detach with Ctrl-A d ; reattach later with:  screen -x orbit
```

`run_rolf.sh` trains grid-192 (aug), grid-128 (aug), and grid-128 (no-aug for
DAVIS) — each with early stopping — then runs GPU inference, assembles the
per-sensor router, and writes `Evaluation_Metrics.xlsx`.

## 3. Individual commands (if you'd rather run pieces)

```bash
# train one model (GPU, early stop)
python3 scripts/train_centernet.py --device cuda --workers 8 \
    --grid 192 --dim 128 --augment --epochs 40 --patience 6 --batch 128 \
    --out models/evt_g192.pt

# inference (GPU)
python3 scripts/infer_centernet.py --device cuda --data-dir OrbitSight_Dataset/Testing_sets \
    --model models/evt_g192.pt --out-dir predictions/g192

# score
python3 Dataloader/evaluate.py --gt-dir OrbitSight_Dataset/Testing_sets \
    --pred-dir predictions/g192 --excel-out Evaluation_Metrics.xlsx
```

## 4. Bring results back

```bash
# from your laptop
rsync -av <user>@rolf.ifi.uzh.ch:~/OrbitalAI/models/            OrbitalAI/models/
rsync -av <user>@rolf.ifi.uzh.ch:~/OrbitalAI/Evaluation_Metrics.xlsx OrbitalAI/
rsync -av <user>@rolf.ifi.uzh.ch:~/OrbitalAI/predictions/       OrbitalAI/predictions/
```

---

## Notes / etiquette (from the group guide)

- **One GPU** unless the box is idle (`CUDA_VISIBLE_DEVICES=0`); the models are
  small (<1 M params) so a single 2080 Ti is plenty.
- Run **`nice -n 15`** and inside **`screen`**.
- Keep the **dataset on `/local/scratch`**, code/results in home.
- Watch memory with `htop -u <user>`; the event arrays are a few GB per sequence
  loaded together (~1–2 GB) — fine on 512 GB, but don't launch many at once.
- No X server on rolf: the visualizer's plots must be written to file (they are),
  never shown.

## GPU knobs worth sweeping (cheap on GPU, were too slow on CPU)

| Flag | Try | Why |
|---|---|---|
| `--grid` | 128 / 192 / 256 | finer localization for small DVX/DAVIS objects |
| `--epochs`/`--patience` | 40 / 6–10 | let early stopping find the sweet spot |
| `--batch` | 128–256 | GPU throughput (lower if OOM) |
| `--dim` | 128 / 192 | model capacity |
| `--augment` | on/off per model | on for EVK4/DVX, off helped DAVIS |
| `--tbins` | 3 / 5 | temporal resolution in the voxel |
