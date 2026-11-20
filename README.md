# HeartPump — EchoNet-Dynamic EF regression scaffold

Minimal runnable scaffold for end-to-end ejection-fraction (EF) regression on EchoNet-Dynamic.

## Data access (no download in repo)

- Owner: Stanford University School of Medicine; 10,030 apical-4-chamber echo videos, routine care 2016–2018, Stanford Hospital.
- Paper: Ouyang et al., Nature 2020.
- Access: Stanford AIMI portal via Redivis (`echonet.github.io/dynamic`, access section); non-commercial Research Use Agreement required.
- Expected layout after download:
  ```
  data/
    FileList.csv        # columns include FileName, Split (TRAIN/VAL/TEST), EF
    VolumeTracings.csv  # ED/ES border coords (segmentation only; not used for EF labels in v1)
    Videos/*.avi
  ```

## DPIO splits

- Split on `FileList.csv` `Split` column, case-insensitive: TRAIN / VAL / TEST (loader upper-cases).
- Official counts: 7465 train / 1288 val / 1277 test (= 10030).
- Split strictly by video (never by clip/frame); never tune on test.
- v1 does NOT replicate upstream tracer-based dropping (videos with <2 traced frames); log effective N vs official counts to quantify the selection-bias delta.

## EF scale

- EF target is PERCENT (0–100): `(EDV-ESV)/EDV * 100`, loaded as float32.
- Report MAE/RMSE in percentage points. Predicting 0–1 fractions is a scale bug (~100x MAE inflation); `evaluate.py` asserts percent scale.

## Scale warning

- v1 is a minimal single-GPU scaffold, NOT the full paper pipeline: squared-percent MSELoss magnitudes are ~1e4x fraction-scale, so watch first-run loss/grad norms (AdamW defaults below are starting points).
- Dataset has no fixed FPS (variable-length videos, native fps); stride-2 sampling covers unequal time windows if fps varies — audit with `cv2.CAP_PROP_FPS` per file if temporal calibration matters.
- GPU verified: RTX 3050 4GB Laptop, torch `2.14.1+cu126`, `cuda_available=True`. Train/eval default to CUDA when available (`--device auto`), with AMP + grad accumulation.
- GPU memory: clips are `(B,3,32,112,112)` float32; 4GB default is `--batch 1 --accum 8` (effective 8), `--num_workers 0` on Windows. If OOM, keep `--batch 1` and lower eval `--num_clips`.

## Setup

```powershell
pip install -r requirements.txt
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"
```

## Stats (per-channel mean/std placeholder)

Computed once over training clips in [0,1] scale AFTER download; stored in `configs/stats.json`:

```powershell
python -m src.train --compute_stats_only --data_root data --stats configs/stats.json
python tests/smoke.py   # synthetic end-to-end check, no download needed
```

Until computed, `src/constants.py` holds `MEAN = [0.5, 0.5, 0.5]`, `STD = [0.5, 0.5, 0.5]` placeholders — replace with measured values and use the SAME train stats at val/test (never Kinetics stats).

## Train / eval

```powershell
python -m src.train --data_root data --filelist data/FileList.csv --stats configs/stats.json --device cuda --batch 1 --accum 8
python -m src.evaluate --data_root data --filelist data/FileList.csv --split TEST --checkpoint checkpoints/best.pt --stats configs/stats.json --device cuda
python -m src.baseline --filelist data/FileList.csv
python -m src.benchmark --filelist data/FileList.csv --split TEST
```

## Reference numbers

- Paper MAE ~4.1 percentage points; human inter-observer variability 7.6–13.9%.
- Model must beat the train-mean baseline MAE to claim signal.

## Known caveats

- FPS varies per video; tracer-drop delta unquantified in v1; LR stability under percent-scale MSE unverified — see synthesis unresolved items.
