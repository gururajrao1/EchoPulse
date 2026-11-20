"""Baseline: predict train-mean EF for every test row; report MAE + EF std."""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filelist", default="data/FileList.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.filelist)
    split_col = [c for c in df.columns if c.lower() == "split"][0]
    ef_col = [c for c in df.columns if c.lower() in ("ef", "ejectionfraction", "ejection_fraction")][0]
    train_ef = df[df[split_col].str.upper() == "TRAIN"][ef_col].astype(float).to_numpy()
    test_ef = df[df[split_col].str.upper() == "TEST"][ef_col].astype(float).to_numpy()

    mean_ef = float(train_ef.mean())
    std_ef = float(train_ef.std())
    pred = np.full_like(test_ef, mean_ef)
    mae = mean_absolute_error(test_ef, pred)
    print(f"train mean EF={mean_ef:.3f} (percent), train std={std_ef:.3f}")
    print(f"mean-baseline TEST MAE={mae:.3f} percentage points (n={len(test_ef)})")
    print("Model must beat this MAE to claim signal; cf. paper ~4.1, human 7.6-13.9.")


if __name__ == "__main__":
    main()
