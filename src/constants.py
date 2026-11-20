"""Placeholder per-channel stats in [0,1] scale.

Replace by running:
  python -m src.train --compute_stats_only --data_root data --stats_out configs/stats.json
and use the SAME train stats at val/test. Never hand-fill real values here.
"""
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]
