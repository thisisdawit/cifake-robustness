"""
Robustness audit.

Takes a trained detector and re-scores it on the same test images after each of
the degradations in `transforms.CONDITIONS`: JPEG re-encoding at four
qualities, a resize round trip, blur, noise, and the resize-then-compress
sequence an image undergoes when posted online.

No training happens here. The question is not how well a detector can be made
to score, but how much of a score it keeps once its input stops being pristine.

    python src/evaluate.py --weights models/small_cnn.weights.h5 \\
        --model small_cnn --root C:/DL_data/cifake
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np


def _bootstrap():
    import pathlib
    import sys

    root = str(pathlib.Path(__file__).resolve().parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)


try:
    from . import data as dataset_tools
    from . import transforms as tf_ops
    from .models import BUILDERS
    from .train import build_file_lists, load_images, subsample
except ImportError:
    _bootstrap()
    from src import data as dataset_tools  # noqa: F401
    from src import transforms as tf_ops
    from src.models import BUILDERS
    from src.train import build_file_lists, load_images, subsample


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    """Mean gap between stated confidence and observed accuracy.

    A detector that is wrong while confident is more dangerous than one that
    hesitates, and accuracy alone cannot distinguish the two.
    """
    conf = np.maximum(probs, 1.0 - probs)
    correct = ((probs >= 0.5).astype(np.float32) == labels).astype(np.float32)
    edges = np.linspace(0.5, 1.0, bins + 1)
    ece = 0.0
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        # The first bin is closed on the left so that a confidence of exactly 0.5 is counted.
        m = ((conf >= lo) if i == 0 else (conf > lo)) & (conf <= hi)
        if m.sum() == 0:
            continue
        ece += (m.mean()) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def score(probs: np.ndarray, labels: np.ndarray) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    pred = (probs >= 0.5).astype(np.float32)
    tp = float(((pred == 1) & (labels == 1)).sum())
    fp = float(((pred == 1) & (labels == 0)).sum())
    fn = float(((pred == 0) & (labels == 1)).sum())
    tn = float(((pred == 0) & (labels == 0)).sum())
    precision = tp / max(tp + fp, 1e-9)
    recall = tp / max(tp + fn, 1e-9)
    return {
        "accuracy": float((pred == labels).mean()),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-9),
        "auc": float(roc_auc_score(labels, probs)),
        "average_precision": float(average_precision_score(labels, probs)),
        "ece": expected_calibration_error(probs, labels),
        "mean_confidence": float(np.maximum(probs, 1 - probs).mean()),
        "fake_recall": recall,
        "real_recall": tn / max(tn + fp, 1e-9),
    }


def provenance(weights: str) -> dict:
    """Identify the exact weights and software behind a result file."""
    import PIL
    import sklearn
    import tensorflow as tf

    return {
        "weights_sha256": hashlib.sha256(Path(weights).read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "scikit_learn": sklearn.__version__,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit a detector under image degradation.")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--model", default="small_cnn",
                    choices=["small_cnn", "resnet50", "efficientnetb0"])
    ap.add_argument("--root", default=None)
    ap.add_argument("--no-dedup", dest="dedup", action="store_false",
                    help="audit against the published split instead of the "
                         "deduplicated one")
    ap.set_defaults(dedup=True)
    ap.add_argument("--limit", type=int, default=0,
                    help="images per class (0 = the whole test set)")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--categories", default=None,
                    help="score only these image categories (a group such as 'vehicles' or a "
                         "comma-separated list)")
    ap.add_argument("--conditions", default=None,
                    help="comma-separated subset of conditions to run")
    ap.add_argument("--save-scores", action="store_true",
                    help="also store per-image probabilities for paired analysis")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow an existing result file to be replaced")
    args = ap.parse_args()

    root = dataset_tools.resolve_root(args.root)
    _, _, test_items, split_stats = build_file_lists(root, args.dedup)

    keep = dataset_tools.parse_categories(args.categories)
    if keep:
        test_items = [it for it in test_items if dataset_tools.category_of(it[0]) in keep]

    if args.limit:
        test_items = subsample(test_items, args.limit)

    print(f"test images: {len(test_items)}")
    images = load_images([p for p, _ in test_items])
    labels = np.asarray([l for _, l in test_items], dtype=np.float32)
    print(f"loaded {images.shape} uint8")

    model = BUILDERS[args.model]()
    model.load_weights(args.weights)
    print(f"loaded weights: {args.weights}")

    conditions = (args.conditions.split(",") if args.conditions
                  else list(tf_ops.CONDITIONS))

    tag = args.tag or Path(args.weights).stem.replace(".weights", "")
    if not args.tag and (args.limit or args.categories or args.conditions):
        tag += "_partial"
    path = Path(args.outdir) / f"audit_{tag}.json"
    if path.exists() and not args.overwrite:
        raise SystemExit(f"{path} exists; pass --overwrite to replace it or --tag to keep both")

    results, scores = {}, {}
    for name in conditions:
        t0 = time.time()
        degraded = (images if name == "clean"
                    else tf_ops.build_degraded_set(images, name))
        probs = model.predict(degraded.astype("float32"),
                              batch_size=args.batch_size, verbose=0).ravel()
        m = score(probs, labels)
        scores[name] = probs.astype(np.float32)
        m["seconds"] = round(time.time() - t0, 1)
        results[name] = m
        print(f"  {name:<14} acc {m['accuracy']:.4f}  f1 {m['f1']:.4f}  "
              f"auc {m['auc']:.4f}  ece {m['ece']:.4f}  "
              f"fake-recall {m['fake_recall']:.4f}  ({m['seconds']}s)")

    clean = results.get("clean")
    record = {
        "tag": tag,
        "model": args.model,
        "weights": args.weights,
        "n_test": len(test_items),
        "dedup": args.dedup,
        "categories": sorted(keep) if keep else "all",
        "split": split_stats,
        "provenance": provenance(args.weights),
        "conditions": results,
    }
    if clean:
        record["drop_from_clean"] = {
            name: {k: round(clean[k] - m[k], 4)
                   for k in ("accuracy", "f1", "auc", "fake_recall")}
            for name, m in results.items() if name != "clean"
        }

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    if args.save_scores:
        np.savez_compressed(out / f"scores_{tag}.npz", labels=labels,
                            files=np.asarray([Path(p).name for p, _ in test_items]), **scores)
    print(f"\nwritten -> {path}")


if __name__ == "__main__":
    main()
