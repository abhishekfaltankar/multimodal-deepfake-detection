"""
Evaluation metrics for the deepfake checker on your own labeled videos.

Folder layout (next to detect.py):
    dataset/real/*.mp4      <- real videos (real person, real voice)
    dataset/fake/*.mp4      <- fake videos (AI / cloned voice)
Aim for ~100 videos total (include reels, noisy and compressed ones).

Run:   python evaluate.py
Needs: pip install scikit-learn   (matplotlib optional, for the confusion-matrix image)

60% of videos are used to pick the best threshold, 40% are held out.
The HOLDOUT numbers are the ones to report (never report numbers from the clips you tuned on).
"""
import csv
import glob
import os
import random
import sys

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)

import detect

ROOT = "dataset"
EXT = (".mp4", ".mov", ".webm", ".mkv", ".avi", ".wav", ".mp3", ".m4a")


def collect():
    items = []
    for name, label in (("real", 0), ("fake", 1)):      # 1 = fake is the "positive" class
        for p in sorted(glob.glob(os.path.join(ROOT, name, "*"))):
            if p.lower().endswith(EXT):
                items.append((p, label))
    return items


def split(labels, tune_frac=0.6, seed=42):
    rng = random.Random(seed)
    tune, hold = [], []
    for c in (0, 1):
        idx = [i for i, l in enumerate(labels) if l == c]
        rng.shuffle(idx)
        cut = int(len(idx) * tune_frac)
        tune += idx[:cut]
        hold += idx[cut:]
    return tune, hold


def metrics(name, y, s, thr):
    pred = (s >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    print(f"\n=== {name} (threshold {thr:.3f}, {len(y)} videos) ===")
    print(f"Accuracy   : {accuracy_score(y, pred):.3f}")
    print(f"Precision  : {precision_score(y, pred, zero_division=0):.3f}   (of videos flagged fake, how many truly are)")
    print(f"Recall     : {recall_score(y, pred, zero_division=0):.3f}   (of real fakes, how many were caught)")
    print(f"Specificity: {spec:.3f}   (of real videos, how many were passed as real)")
    print(f"F1-score   : {f1_score(y, pred, zero_division=0):.3f}")
    if len(set(y)) == 2:
        print(f"ROC-AUC    : {roc_auc_score(y, s):.3f}")
    print("Confusion matrix (rows = actual, cols = predicted):")
    print("              pred REAL  pred FAKE")
    print(f"  actual REAL   {tn:6d}     {fp:6d}")
    print(f"  actual FAKE   {fn:6d}     {tp:6d}")
    return np.array([[tn, fp], [fn, tp]])


def save_matrix_image(cm, path="confusion_matrix.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(install matplotlib to also save a confusion-matrix image)")
        return
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["REAL", "FAKE"])
    ax.set_yticks([0, 1], ["REAL", "FAKE"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (holdout)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")


def main():
    items = collect()
    if len(items) < 20:
        sys.exit("Put at least ~20 videos in dataset/real and dataset/fake (aim for 100).")

    detect.load_models()
    rows = []
    for path, label in items:
        try:
            r = detect.analyze(path)
        except Exception as e:
            print(f"skipped {path}: {e}")
            continue
        rows.append({"path": path, "label": label, "score": r["score"], "verdict": r["verdict"],
                     "audio_score": r["audio"]["score"] if r["audio"] else None,
                     "video_score": r["video"]["score"] if r["video"] else None})
        print(f"{os.path.basename(path):40s} actual={'FAKE' if label else 'REAL'}  "
              f"score={r['score']:.3f}  verdict={r['verdict']}")

    if len({r["label"] for r in rows}) < 2:
        sys.exit("Need both real and fake videos to evaluate.")

    with open("results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\nPer-video scores saved to results.csv")

    y = np.array([r["label"] for r in rows])
    s = np.array([r["score"] for r in rows])

    # how good is each part on its own? (helps you tune detect.WEIGHTS)
    print("\nPer-part ROC-AUC (higher is better):")
    for key, name in (("audio_score", "Voice"), ("video_score", "Face")):
        sub = [(r["label"], r[key]) for r in rows if r[key] is not None]
        labs = {l for l, _ in sub}
        if len(labs) == 2:
            yy = np.array([l for l, _ in sub])
            ss = np.array([v for _, v in sub])
            print(f"  {name}: {roc_auc_score(yy, ss):.3f}  ({len(sub)} videos)")
        else:
            print(f"  {name}: not enough data")

    # 1) default threshold on all videos
    metrics("ALL videos @ default 0.50", y, s, 0.5)

    # 2) tune threshold on 60%, report on the untouched 40%
    tune, hold = split(list(y))
    fpr, tpr, thr = roc_curve(y[tune], s[tune])
    best = float(thr[np.argmax(tpr - fpr)])
    cm = metrics("HOLDOUT @ tuned threshold  <-- report this one", y[hold], s[hold], best)
    save_matrix_image(cm)

    # 3) how often the checker says UNCERTAIN with the current band
    unc = np.array([r["verdict"] == "UNCERTAIN" for r in rows])
    print(f"\nUNCERTAIN verdicts with current band [{detect.REAL_BELOW}, {detect.FAKE_ABOVE}]: {unc.mean():.0%}")
    print(f"Suggested settings in detect.py: REAL_BELOW = {max(best - 0.1, 0):.2f}, "
          f"FAKE_ABOVE = {min(best + 0.1, 1):.2f}")


if __name__ == "__main__":
    main()
