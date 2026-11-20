"""Video loading + clip sampling for EchoNet-Dynamic EF regression.

Conventions (pinned spec):
  L=32 frames, stride s=2, span=(32-1)*2+1=63.
  Frames resized to 112x112 BEFORE clip sampling.
  Short videos use edge-repeat (never zero-pad); corrupt/empty videos raise.
"""
import cv2
import numpy as np

CLIP_LEN = 32
PERIOD = 2
SPAN = (CLIP_LEN - 1) * PERIOD + 1  # 63
SIZE = 112


def load_video(path, size=SIZE):
    """Load AVI to (C=3, T, H, W) float32 in [0, 1]. Grayscale replicated x3."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_CUBIC)
        frames.append(frame)
    cap.release()
    if len(frames) == 0:
        raise ValueError(f"No frames decoded: {path}")
    arr = np.stack(frames, axis=0).astype(np.float32) / 255.0  # (T,H,W,3)
    return np.transpose(arr, (3, 0, 1, 2))  # (3,T,H,W)


def sample_clip(video, start=None, clip_len=CLIP_LEN, period=PERIOD, train=True):
    """Sample one (3, clip_len, H, W) clip. Train: random start; eval: center."""
    _, t, _, _ = video.shape
    span = (clip_len - 1) * period + 1
    if t < clip_len:
        idx = np.linspace(0, t - 1, clip_len).round().astype(int)  # edge-repeat
        return video[:, idx]
    if t < span:
        idx = np.clip(np.arange(clip_len) * period, 0, t - 1)  # edge-repeat tail
        return video[:, idx]
    if start is None:
        if train:
            start = int(np.random.randint(0, t - span + 1))
        else:
            start = (t - span) // 2
    idx = start + np.arange(clip_len) * period
    return video[:, idx]


def sample_multi_clip_starts(t, n_clips=5, clip_len=CLIP_LEN, period=PERIOD):
    """Evenly spaced starts over [0, t-span]; center fallback when short."""
    span = (clip_len - 1) * period + 1
    if t <= span:
        return [max(0, (t - span) // 2)] * n_clips
    if n_clips == 1:
        return [(t - span) // 2]
    return np.linspace(0, t - span, n_clips).round().astype(int).tolist()


def normalize(clip, mean, std):
    """Per-channel normalize; mean/std shape (3,) in [0,1] scale."""
    mean = np.asarray(mean, dtype=np.float32)[:, None, None, None]
    std = np.asarray(std, dtype=np.float32)[:, None, None, None]
    return (clip - mean) / std
