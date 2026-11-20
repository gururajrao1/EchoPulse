"""Synthetic end-to-end smoke test: no EchoNet download needed.

Builds tiny fake AVI videos + FileList.csv in a temp dir, then exercises
loading, clip sampling, normalization, model forward, one training epoch on
CPU, eval, baseline, and the benchmark runner.
"""
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np
import pandas as pd
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.dataset import load_video, normalize, sample_clip
from src.model import build_model

FRAMES, SIZE = 70, 112


def make_avi(path, seed):
    rng = np.random.default_rng(seed)
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 20, (SIZE, SIZE))
    for _ in range(FRAMES):
        w.write((rng.random((SIZE, SIZE, 3)) * 255).astype(np.uint8))
    w.release()


def sh(*args):
    r = subprocess.run([sys.executable, *args], capture_output=True, text=True, cwd=REPO)
    print(r.stdout[-1500:])
    if r.returncode != 0:
        print(r.stderr[-3000:])
        raise SystemExit(f"FAILED: {' '.join(args)}")
    return r.stdout


def main():
    tmp = tempfile.mkdtemp(prefix="echopulse_smoke_")
    vids = os.path.join(tmp, "Videos")
    os.makedirs(vids)
    rows = [("train1", "TRAIN", 65.0), ("train2", "TRAIN", 45.0),
            ("val1", "VAL", 55.0), ("test1", "TEST", 60.0)]
    for name, _, _ in rows:
        make_avi(os.path.join(vids, name + ".avi"), hash(name) % 1000)
    fl = os.path.join(tmp, "FileList.csv")
    pd.DataFrame(rows, columns=["FileName", "Split", "EF"]).to_csv(fl, index=False)
    stats = os.path.join(tmp, "stats.json")
    ckpt = os.path.join(tmp, "best.pt")
    out = os.path.join(tmp, "results.md")

    video = load_video(os.path.join(vids, "train1.avi"))
    assert video.shape[0] == 3 and video.shape[2] == SIZE, video.shape
    clip = sample_clip(video, train=True)
    assert clip.shape == (3, 32, SIZE, SIZE), clip.shape
    clip = normalize(clip, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    model = build_model("r2plus1d_18", pretrained=False)
    assert model(torch.from_numpy(clip).unsqueeze(0)).shape == (1, 1)

    sh("-m", "src.train", "--compute_stats_only", "--filelist", fl,
       "--data_root", vids, "--stats", stats)
    sh("-m", "src.train", "--filelist", fl, "--data_root", vids,
       "--stats", stats, "--batch", "1", "--accum", "1", "--epochs", "1",
       "--device", "cpu", "--num_workers", "0", "--model", "r2plus1d_18")
    assert os.path.exists("checkpoints/best.pt")
    os.replace("checkpoints/best.pt", ckpt)
    sh("-m", "src.evaluate", "--filelist", fl, "--data_root", vids,
       "--split", "TEST", "--checkpoint", ckpt, "--stats", stats,
       "--device", "cpu", "--model", "r2plus1d_18")
    sh("-m", "src.baseline", "--filelist", fl)
    sh("-m", "src.benchmark", "--filelist", fl, "--data_root", vids,
       "--split", "TEST", "--checkpoint", ckpt, "--stats", stats,
       "--models", "r2plus1d_18", "--out", out)
    assert os.path.exists(out)
    print(f"SMOKE OK (tmp={tmp})")


if __name__ == "__main__":
    main()
