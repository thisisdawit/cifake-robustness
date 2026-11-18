"""
Train one detector on CIFAKE and write its metrics to results/.

Metrics are written to a JSON record per run so that reported figures trace
back to a generated file.

    python -m src.train --model small_cnn --root C:/DL_data/cifake --dedup
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import time
from pathlib import Path

import numpy as np

def _bootstrap():
    """Allow these modules to be imported as a package or run as scripts."""
    import pathlib
    import sys

    root = str(pathlib.Path(__file__).resolve().parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)


try:
    from . import data as dataset_tools
except ImportError:
    _bootstrap()
    from src import data as dataset_tools

SEED = 1337
VAL_FRACTION = 0.1


def set_seeds(seed: int = SEED, deterministic: bool = False) -> None:
    """Seed every generator; optionally request deterministic operations (slower)."""
    import tensorflow as tf
    from tensorflow import keras

    keras.utils.set_random_seed(seed)
    if deterministic:
        tf.config.experimental.enable_op_determinism()


def default_tag(model: str, dedup: bool, finetune: bool, harden: bool,
                seed: int = SEED, limit: int = 0) -> str:
    """Name a run from every setting that changes the trained model."""
    tag = model
    if model != "small_cnn":
        tag += "_finetuned" if finetune else "_frozen"
    if not dedup:
        tag += "_published_split"
    if harden:
        tag += "_hardened"
    if seed != SEED:
        tag += f"_s{seed}"
    if limit:
        tag += f"_limit{limit}"
    return tag


def refuse_existing(paths, overwrite: bool) -> None:
    """Stop before training when a run would replace earlier outputs."""
    taken = [str(p) for p in paths if Path(p).exists()]
    if taken and not overwrite:
        raise SystemExit("existing outputs would be replaced: " + ", ".join(taken)
                         + "; pass --overwrite or choose another --tag")


def _digest(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def build_file_lists(root: Path, dedup: bool):
    """Return (train, val, test) lists of (path, label).

    With dedup enabled, any image whose content hash also occurs in the test
    set is dropped from training, and repeated hashes within a split are
    collapsed to a single copy. The official split is left otherwise intact so
    results stay comparable with published work.
    """
    splits = {s: dataset_tools.list_split(root, s) for s in dataset_tools.SPLITS}

    test_items, test_hashes, seen = [], set(), set()
    for cls, paths in splits["test"].items():
        label = dataset_tools.CLASS_DIRS[cls]
        for p in paths:
            h = _digest(p) if dedup else None
            if dedup:
                test_hashes.add(h)
                if h in seen:
                    continue
                seen.add(h)
            test_items.append((p, label))

    train_items, seen = [], set()
    dropped_leak = dropped_dup = 0
    for cls, paths in splits["train"].items():
        label = dataset_tools.CLASS_DIRS[cls]
        for p in paths:
            if dedup:
                h = _digest(p)
                if h in test_hashes:
                    dropped_leak += 1
                    continue
                if h in seen:
                    dropped_dup += 1
                    continue
                seen.add(h)
            train_items.append((p, label))

    rng = random.Random(SEED)
    rng.shuffle(train_items)
    n_val = int(len(train_items) * VAL_FRACTION)
    val_items, train_items = train_items[:n_val], train_items[n_val:]

    stats = {
        "dedup": dedup,
        "n_train": len(train_items),
        "n_val": len(val_items),
        "n_test": len(test_items),
        "dropped_from_train_as_test_leak": dropped_leak,
        "dropped_from_train_as_duplicate": dropped_dup,
    }
    return train_items, val_items, test_items, stats


def subsample(items, per_class: int):
    """Take a balanced subset, preserving class balance and ordering by seed."""
    rng = random.Random(SEED)
    by_class: dict[int, list] = {}
    for item in items:
        by_class.setdefault(item[1], []).append(item)
    out = []
    for label in sorted(by_class):
        group = by_class[label]
        rng.shuffle(group)
        out.extend(group[:per_class])
    rng.shuffle(out)
    return out


_PIXEL_CACHE: dict[tuple, "np.ndarray"] = {}


def load_images(paths, image_size=(32, 32), workers=8):
    """Decode images once into a single uint8 array held in memory.

    The corpus is 120,000 images of 32x32, roughly 368 MB as raw bytes, which
    fits comfortably in memory. Decoding once and training from arrays keeps the
    run compute-bound rather than I/O-bound.
    """
    from concurrent.futures import ThreadPoolExecutor

    from PIL import Image

    out = np.empty((len(paths), *image_size, 3), dtype=np.uint8)
    todo = [(i, p) for i, p in enumerate(paths) if (str(p), image_size) not in _PIXEL_CACHE]

    def decode(job):
        i, path = job
        with Image.open(path) as im:
            arr = np.asarray(im.convert("RGB"))
        if arr.shape[:2] != image_size:
            with Image.open(path) as im:
                arr = np.asarray(im.convert("RGB").resize(image_size[::-1], Image.BICUBIC))
        _PIXEL_CACHE[(str(path), image_size)] = arr
        return i, arr

    if todo:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, arr in pool.map(decode, todo):
                out[i] = arr
    for i, path in enumerate(paths):
        out[i] = _PIXEL_CACHE[(str(path), image_size)]
    return out


def make_dataset(items, batch_size: int, shuffle: bool, image_size=(32, 32),
                 degrade: bool = False, seed: int = SEED, parallel=None):
    """Build a tf.data pipeline over in-memory arrays.

    With `degrade`, each image is randomly degraded every time it is drawn
    (see `transforms.tf_training_degradation`). Only the training set is ever
    built this way; validation and test images stay pristine.
    """
    import tensorflow as tf

    parallel = tf.data.AUTOTUNE if parallel is None else parallel
    try:
        from .transforms import tf_training_degradation
    except ImportError:
        from src.transforms import tf_training_degradation

    paths = [p for p, _ in items]
    labels = np.asarray([l for _, l in items], dtype=np.float32)
    images = load_images(paths, image_size)

    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    if shuffle:
        ds = ds.shuffle(min(len(items), 20000), seed=seed, reshuffle_each_iteration=True)
    if degrade:
        ds = ds.map(tf_training_degradation(), num_parallel_calls=parallel)
    else:
        ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=parallel)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a CIFAKE detector.")
    ap.add_argument("--model", default="small_cnn",
                    choices=["small_cnn", "resnet50", "efficientnetb0"])
    ap.add_argument("--root", default=None)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dedup", action="store_true",
                    help="remove train/test duplicate leakage before training")
    ap.add_argument("--finetune", action="store_true",
                    help="unfreeze the pretrained backbone (transfer models only); "
                         "ImageNet features encode semantics, which may transfer "
                         "poorly to artefact detection unless adapted")
    ap.add_argument("--harden", action="store_true",
                    help="randomly degrade training images (the hardened variant)")
    ap.add_argument("--train-categories", default=None,
                    help="train and validate only on these image categories (a group such as "
                         "'animals' or a comma-separated list); the test set is restricted to the "
                         "same categories, and unseen categories are scored separately with evaluate.py")
    ap.add_argument("--seed", type=int, default=SEED,
                    help="seed for initialisation, shuffling and degradation; the "
                         "train/validation/test split stays fixed regardless")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--modeldir", default="models")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--limit", type=int, default=0,
                    help="train on this many images per class (0 = all); "
                         "for fast end-to-end checks of the pipeline")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow existing weights and results with this tag to be replaced")
    ap.add_argument("--deterministic", action="store_true",
                    help="request repeatable operations and a sequential input pipeline (slower)")
    args = ap.parse_args()

    import tensorflow as tf
    from tensorflow import keras

    try:
        from .models import BUILDERS
    except ImportError:
        from src.models import BUILDERS

    tag = args.tag or default_tag(args.model, args.dedup, args.finetune, args.harden,
                                  args.seed, args.limit)
    refuse_existing([Path(args.modeldir) / f"{tag}.weights.h5",
                     Path(args.outdir) / f"train_{tag}.json"], args.overwrite)

    set_seeds(args.seed, args.deterministic)
    parallel = 1 if args.deterministic else None
    root = dataset_tools.resolve_root(args.root)
    train_items, val_items, test_items, split_stats = build_file_lists(root, args.dedup)

    keep = dataset_tools.parse_categories(args.train_categories)
    if keep:
        def _only(items):
            return [it for it in items if dataset_tools.category_of(it[0]) in keep]
        train_items, val_items, test_items = _only(train_items), _only(val_items), _only(test_items)
        split_stats.update({"train_categories": sorted(keep), "n_train": len(train_items),
                            "n_val": len(val_items), "n_test": len(test_items)})

    if args.limit:
        train_items = subsample(train_items, args.limit)
        val_items = subsample(val_items, max(1, args.limit // 5))
        test_items = subsample(test_items, max(1, args.limit // 5))
        split_stats.update({
            "subsampled_per_class": args.limit,
            "n_train": len(train_items),
            "n_val": len(val_items),
            "n_test": len(test_items),
        })
    print(json.dumps(split_stats, indent=2))

    train_ds = make_dataset(train_items, args.batch_size, shuffle=True,
                            degrade=args.harden, seed=args.seed, parallel=parallel)
    val_ds = make_dataset(val_items, args.batch_size, shuffle=False, parallel=parallel)
    test_ds = make_dataset(test_items, args.batch_size, shuffle=False, parallel=parallel)

    build_kwargs = {}
    if args.model != "small_cnn":
        build_kwargs["trainable_backbone"] = args.finetune
    model = BUILDERS[args.model](**build_kwargs)
    model.compile(
        optimizer=keras.optimizers.Adam(args.lr),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )
    model.summary()

    Path(args.modeldir).mkdir(parents=True, exist_ok=True)
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    weights_path = str(Path(args.modeldir) / f"{tag}.weights.h5")

    # Per-epoch metrics are streamed to CSV as training proceeds: the file can be
    # opened while a run is in progress, and it is the source for the training
    # curves reported later.
    progress_csv = Path(args.outdir) / f"progress_{tag}.csv"

    callbacks = [
        keras.callbacks.CSVLogger(str(progress_csv), append=False),
        keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=4,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
        keras.callbacks.ModelCheckpoint(weights_path, monitor="val_auc", mode="max",
                                        save_best_only=True, save_weights_only=True),
    ]

    t0 = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                        callbacks=callbacks, verbose=2)
    train_seconds = time.time() - t0

    evaluation = model.evaluate(test_ds, return_dict=True, verbose=0)

    trainable = int(sum(int(w.shape.num_elements() or 0) for w in model.trainable_weights))
    total = int(sum(int(w.shape.num_elements() or 0) for w in model.weights))

    record = {
        "tag": tag,
        "parameters": {"trainable": trainable, "total": total},
        "model": args.model,
        "hyperparameters": {
            "epochs_requested": args.epochs,
            "epochs_run": len(history.history["loss"]),
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "seed": args.seed,
            "split_seed": SEED,
            "val_fraction": VAL_FRACTION,
            "hardened": args.harden,
            "train_categories": args.train_categories,
            "limit_per_class": args.limit,
            "finetune_backbone": args.finetune,
        },
        "full_run": args.limit == 0,
        "split": split_stats,
        "train_seconds": round(train_seconds, 1),
        "clean_test": {k: float(v) for k, v in evaluation.items()},
        "history": {k: [float(x) for x in v] for k, v in history.history.items()},
        "environment": {
            "tensorflow": tf.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "gpu": [d.name for d in tf.config.list_physical_devices("GPU")],
        },
        "weights": weights_path,
    }
    record["progress_csv"] = str(progress_csv)
    out = Path(args.outdir) / f"train_{tag}.json"
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")

    f1 = 2 * evaluation["precision"] * evaluation["recall"] / max(
        evaluation["precision"] + evaluation["recall"], 1e-9)
    print(f"\n{tag}: accuracy={evaluation['accuracy']:.4f} "
          f"precision={evaluation['precision']:.4f} recall={evaluation['recall']:.4f} "
          f"f1={f1:.4f} auc={evaluation['auc']:.4f}")
    print(f"written -> {out}")


if __name__ == "__main__":
    main()
