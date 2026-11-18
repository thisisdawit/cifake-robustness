"""
Analyses behind the robustness audit. Each is a sub-command:

    spectrum   radially averaged power spectra and the single-band spectral probe
    content    accuracy per image category, clean and after an upload-style degradation
    explain    Grad-CAM maps and their stability when the image is degraded
    external   validation on GenImage generators brought to CIFAKE's format
    curves     ROC and precision-recall curves, clean and degraded

    python -m src.analysis spectrum --root /path/to/cifake
    python -m src.analysis content --root /path/to/cifake
    python -m src.analysis explain --root /path/to/cifake
    python -m src.analysis external --root /path/to/GenImage --weights-dir models
    python -m src.analysis curves --root /path/to/cifake

Every sub-command writes a JSON record to results/; `python -m src.analysis <name> --help`
lists its options.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

try:
    from . import data as dataset_tools
    from . import transforms as tf_ops
    from .evaluate import score
    from .models import BUILDERS
    from .train import build_file_lists, load_images, subsample
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src import data as dataset_tools
    from src import transforms as tf_ops
    from src.evaluate import score
    from src.models import BUILDERS
    from src.train import build_file_lists, load_images, subsample


# ======================================================================
# Frequency analysis
# ======================================================================

#: Rings at or above this radius form the high-frequency band. On a 32x32
#: image Nyquist is radius 16, so this is the upper half of the spectrum.
HF_START = 8


def _rings(size: int):
    c = size // 2
    y, x = np.indices((size, size))
    r = np.rint(np.hypot(y - c, x - c)).astype(int)
    radii = np.arange(1, c + 1)
    masks = [(r == k) for k in radii]
    return radii, masks


def radial_power(images: np.ndarray) -> np.ndarray:
    """Per-image mean power in each frequency ring, shape (N, size // 2).

    Luminance is used so colour statistics do not enter. Each image has its
    mean removed and is multiplied by a Hann window before the transform, so
    the hard image border does not leak energy into every frequency.
    """
    x = images.astype(np.float32)
    y = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
    y -= y.mean(axis=(1, 2), keepdims=True)
    size = y.shape[1]
    win = np.outer(np.hanning(size), np.hanning(size)).astype(np.float32)
    f = np.fft.fftshift(np.fft.fft2(y * win), axes=(1, 2))
    p = (f.real ** 2 + f.imag ** 2).reshape(len(y), -1)
    _, masks = _rings(size)
    return np.stack([p[:, m.ravel()].mean(axis=1) for m in masks], axis=1)


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC AUC with label 1 as positive (Mann-Whitney form, ties averaged)."""
    from scipy.stats import rankdata

    ranks = rankdata(scores)
    pos = labels == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _fit_probe(feats: np.ndarray, labels: np.ndarray):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(feats, labels)


def _probe_scores(model, feats, labels):
    p = model.predict_proba(feats)[:, 1]
    pred = (p >= 0.5).astype(int)
    return {
        "accuracy": float((pred == labels).mean()),
        "auc": auc(p, labels),
        "fake_recall": float(pred[labels == 1].mean()),
        "real_recall": float(1 - pred[labels == 0].mean()),
    }


def _correlate(spec: dict, audit_path: Path) -> dict:
    from scipy.stats import pearsonr, spearmanr

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    conds = [c for c in spec if c in audit["conditions"]]
    sep = np.array([spec[c]["hf_separability"] for c in conds])
    out = {"tag": audit.get("tag", audit_path.stem), "conditions": conds}
    for metric in ("accuracy", "fake_recall", "auc"):
        det = np.array([audit["conditions"][c][metric] for c in conds])
        out[metric] = {
            "pearson_r": float(pearsonr(sep, det)[0]),
            "spearman_rho": float(spearmanr(sep, det)[0]),
        }
    return out


def spectrum_main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=None)
    ap.add_argument("--per-class", type=int, default=0,
                    help="balanced test subset per class; 0 uses the full test set")
    ap.add_argument("--probe-train-per-class", type=int, default=10000,
                    help="clean training images per class used to fit the spectral probe")
    ap.add_argument("--audit", nargs="*", default=[],
                    help="audit result files to relate spectral separability to")
    ap.add_argument("--out", default="results/spectrum.json")
    args = ap.parse_args()

    t0 = time.time()
    root = dataset_tools.resolve_root(args.root)
    train_items, _, test_items, stats = build_file_lists(root, dedup=True)
    if args.per_class:
        test_items = subsample(test_items, args.per_class)
    probe_items = subsample(train_items, args.probe_train_per_class)

    test_x = load_images([p for p, _ in test_items])
    test_y = np.array([y for _, y in test_items])
    probe_x = load_images([p for p, _ in probe_items])
    probe_y = np.array([y for _, y in probe_items])
    print(f"test {len(test_y)}  probe-train {len(probe_y)}  ({time.time() - t0:.0f}s)", flush=True)

    radii, _ = _rings(test_x.shape[1])
    hf = radii >= HF_START
    bands = {"full": np.ones_like(hf), "low": ~hf, "high": hf}

    train_logp = np.log10(radial_power(probe_x) + 1e-6)
    probes = {b: _fit_probe(train_logp[:, m], probe_y) for b, m in bands.items()}

    results = {}
    for cond in tf_ops.CONDITIONS:
        xd = tf_ops.build_degraded_set(test_x, cond)
        power = radial_power(xd)
        logp = np.log10(power + 1e-6)
        real, fake = logp[test_y == 0], logp[test_y == 1]
        per_r = [auc(logp[:, i], test_y) for i in range(len(radii))]
        sep = [max(a, 1 - a) for a in per_r]
        ring_energy = power * np.array([m.sum() for m in _rings(test_x.shape[1])[1]])
        hf_frac = ring_energy[:, hf].sum(1) / ring_energy.sum(1)
        results[cond] = {
            "mean_log10_power_real": real.mean(0).tolist(),
            "mean_log10_power_fake": fake.mean(0).tolist(),
            "gap_fake_minus_real": (fake.mean(0) - real.mean(0)).tolist(),
            "per_radius_auc": per_r,
            "hf_separability": float(np.mean(np.array(sep)[hf])),
            "lf_separability": float(np.mean(np.array(sep)[~hf])),
            "hf_energy_fraction_real": float(hf_frac[test_y == 0].mean()),
            "hf_energy_fraction_fake": float(hf_frac[test_y == 1].mean()),
            "hf_energy_fraction_auc": auc(hf_frac, test_y),
            "probe": {b: _probe_scores(probes[b], logp[:, m], test_y) for b, m in bands.items()},
        }
        r = results[cond]
        print(f"{cond:<12} hf-sep {r['hf_separability']:.3f}  lf-sep {r['lf_separability']:.3f}  "
              f"probe acc {r['probe']['full']['accuracy']:.3f}  "
              f"fake-recall {r['probe']['full']['fake_recall']:.3f}", flush=True)

    out = {
        "n_test": int(len(test_y)),
        "n_probe_train": int(len(probe_y)),
        "dedup": True,
        "split": stats,
        "radii": radii.tolist(),
        "hf_start": HF_START,
        "window": "hann",
        "channel": "luminance (ITU-R BT.601)",
        "conditions": results,
        "detector_correlation": [_correlate(results, Path(a)) for a in args.audit],
        "seconds": round(time.time() - t0, 1),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", args.out)

# ======================================================================
# Accuracy per image category
# ======================================================================

def architecture_of(tag: str) -> str:
    for name in ("efficientnetb0", "resnet50", "small_cnn"):
        if tag.startswith(name):
            return name
    raise ValueError(f"cannot infer the architecture of {tag!r}")


def per_category(probs: np.ndarray, labels: np.ndarray, cats: np.ndarray) -> dict:
    out = {}
    for c in dataset_tools.CATEGORIES:
        m = cats == c
        s = score(probs[m], labels[m])
        out[c] = {k: round(s[k], 4) for k in ("accuracy", "auc", "fake_recall", "real_recall")}
        out[c]["n"] = int(m.sum())
    return out


def content_main() -> None:
    ap = argparse.ArgumentParser(description="Break detector accuracy down by image category.")
    ap.add_argument("--root", default=None)
    ap.add_argument("--models", nargs="+", required=True,
                    help="weight tags under --modeldir, e.g. small_cnn efficientnetb0_finetuned")
    ap.add_argument("--modeldir", default="models")
    ap.add_argument("--conditions", default="clean,social_q50")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--out", default="results/content.json")
    args = ap.parse_args()

    root = dataset_tools.resolve_root(args.root)
    _, _, test_items, _ = build_file_lists(root, True)
    images = load_images([p for p, _ in test_items])
    labels = np.asarray([l for _, l in test_items], dtype=np.float32)
    cats = np.asarray([dataset_tools.category_of(p) for p, _ in test_items])
    print(f"test images: {len(test_items)}")

    conditions = args.conditions.split(",")
    degraded = {c: images if c == "clean" else tf_ops.build_degraded_set(images, c)
                for c in conditions}

    out_path = Path(args.out)
    record = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    for tag in args.models:
        model = BUILDERS[architecture_of(tag)]()
        model.load_weights(str(Path(args.modeldir) / f"{tag}.weights.h5"))
        record[tag] = {}
        for c in conditions:
            probs = model.predict(degraded[c].astype("float32"),
                                  batch_size=args.batch_size, verbose=0).ravel()
            pc = per_category(probs, labels, cats)
            accs = [v["accuracy"] for v in pc.values()]
            record[tag][c] = {"per_category": pc,
                              "spread": round(max(accs) - min(accs), 4),
                              "worst": min(pc, key=lambda k: pc[k]["accuracy"]),
                              "best": max(pc, key=lambda k: pc[k]["accuracy"])}
            print(f"  {tag:<32} {c:<11} spread {record[tag][c]['spread']:.3f}  "
                  f"worst {record[tag][c]['worst']}  best {record[tag][c]['best']}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"\nwritten -> {out_path}")

# ======================================================================
# Grad-CAM explanations under degradation
# ======================================================================

def cam_model(model):
    from tensorflow import keras

    last = [l for l in model.layers if l.__class__.__name__ == "Activation"][-1]
    head = model.get_layer("p_fake")
    return keras.Model(model.inputs, [last.output, head.input]), head


def grad_cam(cam, images: np.ndarray, sign: np.ndarray | None = None, batch: int = 256):
    """Return Grad-CAM maps and logits.

    `sign` fixes the explained class per image (+1 generated, -1 photograph). Without
    it, each image explains its own predicted class. The logit is computed directly
    from the output layer, so saturated probabilities keep a non-zero gradient.
    """
    import tensorflow as tf

    cmodel, head = cam
    maps, logits = [], []
    for i in range(0, len(images), batch):
        x = tf.convert_to_tensor(images[i:i + batch].astype("float32"))
        with tf.GradientTape() as tape:
            feats, hidden = cmodel(x, training=False)
            logit = tf.squeeze(tf.matmul(hidden, tf.convert_to_tensor(head.kernel))
                               + tf.convert_to_tensor(head.bias), axis=-1)
            s = (tf.where(logit >= 0, 1.0, -1.0) if sign is None
                 else tf.constant(sign[i:i + batch], tf.float32))
            target = s * logit
        grads = tape.gradient(target, feats)
        weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
        maps.append(tf.nn.relu(tf.reduce_sum(weights * feats, axis=-1)).numpy())
        logits.append(logit.numpy())
    return np.concatenate(maps), np.concatenate(logits)


def _normalise(m: np.ndarray) -> np.ndarray:
    s = m.reshape(len(m), -1).sum(axis=1)[:, None, None]
    return m / np.maximum(s, 1e-12)


def stability(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.reshape(len(a), -1)
    b = b.reshape(len(b), -1)
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    ok = den > 1e-12
    r = np.full(len(a), np.nan)
    r[ok] = (a[ok] * b[ok]).sum(1) / den[ok]
    return r


def border_share(m: np.ndarray) -> np.ndarray:
    n = _normalise(m)
    ring = np.ones(m.shape[1:], bool)
    ring[1:-1, 1:-1] = False
    return n[:, ring].sum(axis=1)


def explain_main() -> None:
    ap = argparse.ArgumentParser(description="Grad-CAM stability of the small CNN.")
    ap.add_argument("--root", default=None)
    ap.add_argument("--models", nargs="+", default=["small_cnn", "small_cnn_hardened"])
    ap.add_argument("--modeldir", default="models")
    ap.add_argument("--conditions", default="resize_0.50,jpeg_q50,social_q50")
    ap.add_argument("--limit", type=int, default=1000, help="test images per class")
    ap.add_argument("--examples", type=int, default=4)
    ap.add_argument("--out", default="results/explain.json")
    args = ap.parse_args()

    root = dataset_tools.resolve_root(args.root)
    _, _, test_items, _ = build_file_lists(root, True)
    test_items = subsample(test_items, args.limit)
    images = load_images([p for p, _ in test_items])
    labels = np.asarray([l for _, l in test_items], dtype=np.float32)
    conditions = args.conditions.split(",")
    degraded = {c: tf_ops.build_degraded_set(images, c) for c in conditions}
    print(f"images: {len(images)}")

    # Fixed example set: the first fake images in the subsample.
    ex = np.flatnonzero(labels == 1)[:args.examples]
    record = {"n": len(images), "conditions": conditions, "uniform_border_share": 28 / 64,
              "models": {}, "examples": {"images": images[ex].tolist(),
                                         "resize_0.50": degraded["resize_0.50"][ex].tolist()
                                         if "resize_0.50" in degraded else None}}
    for tag in args.models:
        model = BUILDERS["small_cnn"]()
        model.load_weights(str(Path(args.modeldir) / f"{tag}.weights.h5"))
        cm = cam_model(model)
        clean, clean_logit = grad_cam(cm, images)
        target = np.where(clean_logit >= 0, 1.0, -1.0)
        empty = clean.reshape(len(clean), -1).sum(1) == 0
        entry = {"empty_clean_maps": int(empty.sum()),
                 "border_share_clean": {
                    "fake": float(np.mean(border_share(clean[(labels == 1) & ~empty]))),
                    "real": float(np.mean(border_share(clean[(labels == 0) & ~empty])))},
                 "stability": {},
                 "mean_map_clean_fake": _normalise(clean[labels == 1]).mean(0).tolist(),
                 "example_maps": {"clean": clean[ex].tolist()}}
        for c in conditions:
            # The degraded map explains the class predicted for the clean image, so a
            # changed prediction does not change the question asked of the map.
            d, d_logit = grad_cam(cm, degraded[c], sign=target)
            r = stability(clean, d)
            flipped = (d_logit >= 0) != (clean_logit >= 0)
            fake = labels == 1
            entry["stability"][c] = {"fake": float(np.nanmean(r[fake])),
                                     "real": float(np.nanmean(r[labels == 0])),
                                     "all": float(np.nanmean(r)),
                                     "valid_maps": int(np.isfinite(r).sum()),
                                     "fake_prediction_changed": float(flipped[fake].mean()),
                                     "fake_changed": float(np.nanmean(r[fake & flipped]))
                                     if (fake & flipped).any() else None,
                                     "fake_unchanged": float(np.nanmean(r[fake & ~flipped]))
                                     if (fake & ~flipped).any() else None}
            if c == "resize_0.50":
                entry["example_maps"]["resize_0.50"] = d[ex].tolist()
            print(f"  {tag:<22} {c:<12} stability fake {entry['stability'][c]['fake']:.3f}  "
                  f"real {entry['stability'][c]['real']:.3f}")
        print(f"  {tag:<22} border share fake {entry['border_share_clean']['fake']:.3f}  "
              f"real {entry['border_share_clean']['real']:.3f}")
        record["models"][tag] = entry

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record), encoding="utf-8")
    print(f"\nwritten -> {out}")

# ======================================================================
# External validation on GenImage
# ======================================================================

SIZE = 32
CIFAKE_JPEG_QUALITY = 75
SEED = 1337
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def find_generators(root: Path, split: str = "val", max_depth: int = 5) -> dict[str, Path]:
    """Map generator name -> its split directory holding `ai/` and `nature/`.

    GenImage releases nest these differently, so subsets are found by content:
    any directory containing both `ai` and `nature` is one generator's split,
    named after the nearest ancestor that is not a split name.
    """
    found = {}
    frontier = [(root, 0)]
    while frontier:
        cur, depth = frontier.pop(0)
        try:
            subdirs = sorted(e.name for e in os.scandir(cur) if e.is_dir())
        except OSError:
            continue
        if "ai" in subdirs and "nature" in subdirs:
            name = cur.name
            if name.lower() in {"val", "train", "test"}:
                if name.lower() != split:
                    continue
                name = cur.parent.name
            if name in found:
                raise SystemExit(f"two directories claim generator {name!r}: {found[name]} and {cur}")
            found[name] = cur
            continue
        if depth < max_depth:
            frontier.extend((cur / s, depth + 1) for s in subdirs)
    return dict(sorted(found.items()))


def to_cifake_format(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        s = min(w, h)
        left, top = (w - s) // 2, (h - s) // 2
        im = im.crop((left, top, left + s, top + s)).resize((SIZE, SIZE), Image.BICUBIC)
        arr = np.asarray(im)
    return tf_ops.jpeg(arr, CIFAKE_JPEG_QUALITY)


def _list_images(d: Path) -> list[Path]:
    with os.scandir(d) as it:
        return sorted(Path(e.path) for e in it
                      if e.is_file() and Path(e.name).suffix.lower() in IMAGE_SUFFIXES)


def load_subset(split_dir: Path, per_class: int, workers: int = 8):
    from concurrent.futures import ThreadPoolExecutor

    rng = random.Random(SEED)
    items = []
    for cls, label in (("nature", 0), ("ai", 1)):
        files = _list_images(split_dir / cls)
        rng.shuffle(files)
        items.extend((f, label) for f in files[:per_class])
    with ThreadPoolExecutor(max_workers=workers) as pool:
        images = np.stack(list(pool.map(lambda it: to_cifake_format(it[0]), items)))
    labels = np.array([l for _, l in items], dtype=np.float32)
    return images, labels, [f.relative_to(split_dir).as_posix() for f, _ in items]


def architecture_for(tag: str) -> str:
    for arch in ("small_cnn", "resnet50", "efficientnetb0"):
        if tag.startswith(arch):
            return arch
    raise KeyError(f"cannot infer architecture from {tag!r}")


def external_main() -> None:
    ap = argparse.ArgumentParser(description="Score detectors on unseen generators.")
    ap.add_argument("--root", required=True, help="directory containing the GenImage subsets")
    ap.add_argument("--weights-dir", required=True)
    ap.add_argument("--per-class", type=int, default=1000)
    ap.add_argument("--split", default="val", choices=["val", "train", "test"])
    ap.add_argument("--tags", nargs="*", default=None,
                    help="weight tags to score; by default every detector trained on all categories")
    ap.add_argument("--conditions", default="clean,jpeg_q50,resize_0.50,social_q70",
                    help="comma-separated conditions from transforms.CONDITIONS")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--out", default="results/external_genimage.json")
    args = ap.parse_args()

    t0 = time.time()
    gens = find_generators(Path(args.root), args.split)
    if not gens:
        raise SystemExit(f"no GenImage subsets (ai/ + nature/) found under {args.root}")
    print("generators:", list(gens))

    data, sampled = {}, {}
    for name, d in gens.items():
        x, y, sampled[name] = load_subset(d, args.per_class)
        data[name] = (x, y)
        print(f"  {name:<12} {int((y == 0).sum())} nature + {int((y == 1).sum())} ai "
              f"({time.time() - t0:.0f}s)", flush=True)

    conditions = args.conditions.split(",")
    weights = sorted(Path(args.weights_dir).glob("*.weights.h5"))
    if args.tags is not None:
        weights = [w for w in weights if w.name.replace(".weights.h5", "") in set(args.tags)]
    else:
        weights = [w for w in weights if not w.name.startswith("content_")]
    results = {}
    for w in weights:
        tag = w.name.replace(".weights.h5", "")
        model = BUILDERS[architecture_for(tag)]()
        model.load_weights(str(w))
        results[tag] = {}
        for name, (x, y) in data.items():
            results[tag][name] = {}
            for cond in conditions:
                xd = x if cond == "clean" else tf_ops.build_degraded_set(x, cond)
                p = model.predict(xd.astype("float32"), batch_size=args.batch_size, verbose=0).ravel()
                results[tag][name][cond] = score(p, y)
            c = results[tag][name]["clean"]
            print(f"  {tag:<34} {name:<12} acc {c['accuracy']:.3f}  auc {c['auc']:.3f}  "
                  f"fake-recall {c['fake_recall']:.3f}", flush=True)

    out = {
        "dataset": "GenImage (validation split)",
        "protocol": {
            "per_class_per_generator": args.per_class,
            "seed": SEED,
            "preprocessing": f"centre crop, bicubic to {SIZE}x{SIZE}, JPEG q{CIFAKE_JPEG_QUALITY}, both classes",
            "conditions": conditions,
        },
        "generators": {n: {"n": int(len(y)), "dir": str(d)} for n, ((_, y), d) in
                       zip(data, zip(data.values(), gens.values()))},
        "sampled_files": sampled,
        "results": results,
        "seconds": round(time.time() - t0, 1),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", args.out)

# ======================================================================
# ROC and precision-recall curves
# ======================================================================

DETECTORS = {
    "small_cnn": "small_cnn",
    "small_cnn_hardened": "small_cnn",
    "efficientnetb0_finetuned": "efficientnetb0",
    "efficientnetb0_finetuned_hardened": "efficientnetb0",
}
CONDITIONS = ("clean", "social_q50")


def _resample(x, y, n=200):
    """Keep the curve's shape with at most n points, always including the ends."""
    if len(x) <= n:
        return [float(v) for v in x], [float(v) for v in y]
    idx = np.unique(np.linspace(0, len(x) - 1, n).round().astype(int))
    return [float(v) for v in np.asarray(x)[idx]], [float(v) for v in np.asarray(y)[idx]]


def curves_main() -> None:
    from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=None)
    ap.add_argument("--modeldir", default="models")
    ap.add_argument("--out", default="results/curves.json")
    args = ap.parse_args()

    _, _, test_items, _ = build_file_lists(dataset_tools.resolve_root(args.root), dedup=True)
    x = load_images([p for p, _ in test_items])
    y = np.array([l for _, l in test_items])
    degraded = {c: (x if c == "clean" else tf_ops.build_degraded_set(x, c)) for c in CONDITIONS}

    out = {"n_test": int(len(y)), "conditions": list(CONDITIONS), "curves": {}}
    for tag, arch in DETECTORS.items():
        model = BUILDERS[arch]()
        model.load_weights(str(Path(args.modeldir) / f"{tag}.weights.h5"))
        out["curves"][tag] = {}
        for cnd, xd in degraded.items():
            p = model.predict(xd.astype("float32"), batch_size=256, verbose=0).ravel()
            fpr, tpr, _ = roc_curve(y, p)
            prec, rec, _ = precision_recall_curve(y, p)
            fx, fy = _resample(fpr, tpr)
            rx, ry = _resample(rec[::-1], prec[::-1])
            out["curves"][tag][cnd] = {"fpr": fx, "tpr": fy, "recall": rx, "precision": ry,
                                       "roc_auc": float(auc(fpr, tpr)),
                                       "average_precision": float(average_precision_score(y, p))}
            print(f"{tag:<36} {cnd:<11} AUC {out['curves'][tag][cnd]['roc_auc']:.4f}  "
                  f"AP {out['curves'][tag][cnd]['average_precision']:.4f}", flush=True)
    Path(args.out).write_text(json.dumps(out), encoding="utf-8")
    print("wrote", args.out)


COMMANDS = {"spectrum": spectrum_main, "content": content_main, "explain": explain_main,
            "external": external_main, "curves": curves_main}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        raise SystemExit(f"usage: python -m src.analysis {{{','.join(COMMANDS)}}} [options]")
    name = sys.argv.pop(1)
    sys.argv[0] = f"analysis {name}"
    COMMANDS[name]()


if __name__ == "__main__":
    main()
