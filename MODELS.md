# EchoPulse model + algorithm roadmap

Active registry (`src/model.py`): `r2plus1d_18` (default), `mc3_18`, `r3d_18`.
All are Kinetics-400 18-layer video CNNs with `fc` replaced by one EF output (percent 0–100).

## How to train / benchmark each one

```powershell
python -m src.train --model r2plus1d_18 --device cuda --batch 1 --accum 8
python -m src.train --model mc3_18 --device cuda --batch 1 --accum 8
python -m src.train --model r3d_18 --device cuda --batch 1 --accum 8
python -m src.evaluate --model r2plus1d_18 --split TEST --checkpoint checkpoints/best.pt --device cuda
python -m src.benchmark --split TEST
```

`benchmark.py` reports one row per model with a found checkpoint, plus the
train-mean baseline, paper MAE ~4.1, and human range 7.6–13.9.

## Next algorithms (researched, not yet implemented)

1. Temporal ablations: clip length 16 vs 32, stride 1 vs 2, K=5 vs 10 clips.
2. Multi-task EF + ESV/EDV regression; EF-from-volumes consistency check.
3. Segmentation-assisted: VolumeTracings masks as auxiliary supervision.
4. Transformer video backbones (TimeSformer / Video Swin) for long-range motion.
5. Calibration: test-time augmentation + temperature-style bias correction on VAL.

## Interdomain directions

* Same clip-regression setup transfers to fetal echo, lung ultrasound B-lines,
  and echocardiography view classification pre-training.
* Shared pipeline (loader, clip sampler, normalizer, K-clip averaging) already
  isolates domain-specific parts to `FileList` parsing + stats file.
* Rule: never mix domains in one split; pretrain on one, fine-tune on another,
  and report per-domain MAE separately.
