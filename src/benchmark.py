"""Benchmark every model/algo used: train-mean baseline vs R(2+1)D EF regressor.

Runs baseline + multi-clip R(2+1)D eval on the same split, then writes a
comparison table against paper/human references. Requires downloaded data
and a trained checkpoint for the R(2+1)D row; baseline row needs only FileList.
"""
import argparse
import os
import subprocess
import sys


def run(cmd):
    print(f"+ {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    return r.returncode, r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filelist", default="data/FileList.csv")
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--split", default="TEST")
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--stats", default="configs/stats.json")
    ap.add_argument("--num_clips", type=int, default=5)
    ap.add_argument("--models", nargs="*", default=["r2plus1d_18", "mc3_18", "r3d_18"])
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--out", default="results.md")
    args = ap.parse_args()

    base_rows = []
    code, out = run([sys.executable, "-m", "src.baseline", "--filelist", args.filelist])
    base_rows.append(("train-mean baseline", "needs FileList only", out.strip().replace("\n", " | ")))

    for model in args.models:
        ckpt = args.checkpoint if len(args.models) == 1 else os.path.join(
            os.path.dirname(args.checkpoint), f"{model}.pt")
        if os.path.exists(ckpt) and os.path.exists(args.filelist):
            code, out = run([
                sys.executable, "-m", "src.evaluate",
                "--data_root", args.data_root, "--filelist", args.filelist,
                "--split", args.split, "--checkpoint", ckpt,
                "--stats", args.stats, "--num_clips", str(args.num_clips),
                "--model", model, "--device", args.device,
            ])
            base_rows.append((model, f"multi-clip K={args.num_clips}", out.strip().replace("\n", " | ")))
        else:
            base_rows.append((model, "NOT RUN — missing checkpoint/data", ""))

    lines = [
        "# EchoPulse benchmark",
        "",
        f"Split: {args.split}, clips: {args.num_clips}",
        "",
        "| Model / algo | Setup | Result |",
        "|---|---|---|",
    ]
    for name, setup, result in base_rows:
        lines.append(f"| {name} | {setup} | {result} |")
    lines += [
        "| Ouyang et al. (paper) | full heartbeat averaging | MAE ~4.1 pct pts |",
        "| Human experts | inter-observer | 7.6–13.9 pct pts |",
        "",
        "Model must beat the train-mean baseline to claim signal.",
        "",
    ]
    with open(args.out, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
