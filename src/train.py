"""Train EF regressor: MSE on percent-scale EF, val MAE, random-start aug."""
import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .constants import MEAN, STD
from .dataset import CLIP_LEN, PERIOD, load_video, normalize, sample_clip
from .model import build_model


class EFClipDataset(Dataset):
    def __init__(self, filelist, data_root, split, mean, std, train):
        df = pd.read_csv(filelist)
        split_col = [c for c in df.columns if c.lower() == "split"][0]
        df = df[df[split_col].str.upper() == split.upper()].reset_index(drop=True)
        name_col = [c for c in df.columns if c.lower() in ("filename", "file_name", "video")][0]
        ef_col = [c for c in df.columns if c.lower() in ("ef", "ejectionfraction", "ejection_fraction")][0]
        self.rows = list(zip(df[name_col].astype(str), df[ef_col].astype(float)))
        self.data_root = data_root
        self.mean, self.std, self.train = mean, std, train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        name, ef = self.rows[i]
        path = name if os.path.splitext(name)[1] else name + ".avi"
        path = path if os.path.isabs(path) else os.path.join(self.data_root, path)
        video = load_video(path)
        clip = sample_clip(video, train=self.train)  # random start aug when train
        clip = normalize(clip, self.mean, self.std)
        return torch.from_numpy(clip), torch.tensor(float(ef), dtype=torch.float32)


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate_mae(model, loader, device):
    model.eval()
    abs_err, n = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).squeeze(1).cpu()
            abs_err += (pred - y).abs().sum().item()
            n += len(y)
    return abs_err / max(n, 1)


def compute_train_stats(filelist, data_root, max_videos=200):
    """Mean/std over train clips in [0,1]; run post-download, save to stats json."""
    import pandas as pd

    df = pd.read_csv(filelist)
    split_col = [c for c in df.columns if c.lower() == "split"][0]
    names = df[df[split_col].str.upper() == "TRAIN"].iloc[:max_videos]
    name_col = [c for c in df.columns if c.lower() in ("filename", "file_name", "video")][0]
    sum_, sq, cnt = np.zeros(3), np.zeros(3), 0
    for name in names[name_col].astype(str):
        p = name if os.path.splitext(name)[1] else name + ".avi"
        p = p if os.path.isabs(p) else os.path.join(data_root, p)
        v = load_video(p)
        c = sample_clip(v, train=False)
        sum_ += c.mean(axis=(1, 2, 3))
        sq += (c ** 2).mean(axis=(1, 2, 3))
        cnt += 1
    mean = sum_ / cnt
    std = np.sqrt(np.maximum(sq / cnt - mean ** 2, 1e-12))
    return mean.tolist(), std.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--filelist", default="data/FileList.csv")
    ap.add_argument("--stats", default="configs/stats.json")
    ap.add_argument("--batch", type=int, default=1)  # RTX 3050 4GB: per-step batch that fits (3,32,112,112)
    ap.add_argument("--accum", type=int, default=8)  # effective batch = batch*accum = 8
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--model", default="r2plus1d_18", choices=["r2plus1d_18", "mc3_18", "r3d_18"])
    ap.add_argument("--num_workers", type=int, default=0)  # Windows + laptop GPU: 0 avoids shm/fork issues
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compute_stats_only", action="store_true")
    ap.add_argument("--max_stats_videos", type=int, default=200)
    args = ap.parse_args()

    if args.compute_stats_only:
        mean, std = compute_train_stats(args.filelist, args.data_root, args.max_stats_videos)
        os.makedirs(os.path.dirname(args.stats) or ".", exist_ok=True)
        json.dump({"mean": mean, "std": std}, open(args.stats, "w"), indent=2)
        print(f"Wrote {args.stats}: mean={mean} std={std}")
        return

    set_seed(args.seed)
    mean, std = MEAN, STD
    if os.path.exists(args.stats):
        s = json.load(open(args.stats))
        mean, std = s["mean"], s["std"]
    else:
        print(f"WARNING: {args.stats} missing; using placeholder MEAN/STD — compute from data first.")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False")
    device = args.device if args.device in ("cuda", "cpu") else ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)} (cuda {torch.version.cuda})")
    else:
        print("Using CPU (CUDA not selected/available)")
    train_ds = EFClipDataset(args.filelist, args.data_root, "TRAIN", mean, std, train=True)
    val_ds = EFClipDataset(args.filelist, args.data_root, "VAL", mean, std, train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers, pin_memory=(device == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch, num_workers=args.num_workers, pin_memory=(device == "cuda"))

    model = build_model(name=args.model).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    loss_fn = nn.MSELoss()  # EF in percent; MAE reported as metric only
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    best, patience, wait = float("inf"), 8, 0

    for epoch in range(args.epochs):
        model.train()
        opt.zero_grad()
        for step, (x, y) in enumerate(tqdm(train_loader, desc=f"epoch {epoch}")):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                loss = loss_fn(model(x).squeeze(1), y) / args.accum
            scaler.scale(loss).backward()
            if (step + 1) % args.accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
        val_mae = evaluate_mae(model, val_loader, device)
        sched.step(val_mae)
        print(f"epoch {epoch}: val MAE={val_mae:.3f} pct pts")
        os.makedirs("checkpoints", exist_ok=True)
        if val_mae < best:
            best, wait = val_mae, 0
            torch.save(model.state_dict(), "checkpoints/best.pt")
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stop at epoch {epoch}, best val MAE={best:.3f}")
                break


if __name__ == "__main__":
    main()
