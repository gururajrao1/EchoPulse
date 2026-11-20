"""Evaluate: multi-clip average EF per video, MAE/RMSE in percentage points."""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .constants import MEAN, STD
from .dataset import load_video, normalize, sample_clip
from .dataset import CLIP_LEN, PERIOD
from .model import build_model


@torch.no_grad()
def predict_ef(model, video, mean, std, device, n_clips=5):
    from .dataset import sample_multi_clip_starts

    _, t, _, _ = video.shape
    starts = sample_multi_clip_starts(t, n_clips)
    preds = []
    model.eval()
    for s in starts:
        clip = sample_clip(video, start=int(s), train=False)
        clip = normalize(clip, mean, std)
        x = torch.from_numpy(clip).unsqueeze(0).to(device)
        preds.append(model(x).squeeze().cpu().item())
    return float(np.mean(preds))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--filelist", default="data/FileList.csv")
    ap.add_argument("--split", default="TEST")
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--stats", default="configs/stats.json")
    ap.add_argument("--num_clips", type=int, default=5)
    ap.add_argument("--model", default="r2plus1d_18", choices=["r2plus1d_18", "mc3_18", "r3d_18"])
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = ap.parse_args()

    mean, std = MEAN, STD
    if os.path.exists(args.stats):
        s = json.load(open(args.stats))
        mean, std = s["mean"], s["std"]

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False")
    device = args.device if args.device in ("cuda", "cpu") else ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)} (cuda {torch.version.cuda})")
    model = build_model(name=args.model, pretrained=False).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    df = pd.read_csv(args.filelist)
    split_col = [c for c in df.columns if c.lower() == "split"][0]
    name_col = [c for c in df.columns if c.lower() in ("filename", "file_name", "video")][0]
    ef_col = [c for c in df.columns if c.lower() in ("ef", "ejectionfraction", "ejection_fraction")][0]
    sub = df[df[split_col].str.upper() == args.split.upper()].reset_index(drop=True)

    y_true, y_pred = [], []
    for _, row in sub.iterrows():
        name = str(row[name_col])
        p = name if os.path.splitext(name)[1] else name + ".avi"
        p = p if os.path.isabs(p) else os.path.join(args.data_root, p)
        try:
            video = load_video(p)
        except (IOError, ValueError) as e:
            print(f"Skip {p}: {e}")
            continue
        pred = predict_ef(model, video, mean, std, device, args.num_clips)
        if not (0.0 <= pred <= 100.0):
            raise ValueError(f"Scale check failed: pred={pred:.3f} outside [0,100]; model must output EF percent, not 0-1 fraction.")
        y_true.append(float(row[ef_col]))
        y_pred.append(pred)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    print(f"{args.split}: n={len(y_true)} MAE={mae:.3f} RMSE={rmse:.3f} (percentage points)")
    print(f"References: paper MAE ~4.1; human variability 7.6-13.9.")


if __name__ == "__main__":
    main()
