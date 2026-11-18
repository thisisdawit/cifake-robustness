"""
Export Small CNN weights for the browser demo.

Batch normalisation is folded into the preceding convolution, so the browser
only evaluates convolutions, ReLU, pooling and two dense layers. Writes one
float32 binary per model and a manifest describing the layout, plus reference
inputs and Keras outputs used to check the browser implementation.

    python demo/export_weights.py --root ./data/cifake
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "demo" / "web"
MODELS = {"clean": "small_cnn", "hardened": "small_cnn_hardened"}


def fold(model):
    from tensorflow.keras import layers

    arrays, layout = [], []
    pending = None
    for layer in model.layers:
        if isinstance(layer, layers.Conv2D):
            pending = layer.get_weights()[0].astype(np.float64)        # (3, 3, cin, cout)
        elif isinstance(layer, layers.BatchNormalization):
            gamma, beta, mean, var = (w.astype(np.float64) for w in layer.get_weights())
            scale = gamma / np.sqrt(var + layer.epsilon)
            k = pending * scale
            b = beta - mean * scale
            arrays += [k, b]
            layout.append({"type": "conv", "cin": k.shape[2], "cout": k.shape[3]})
            pending = None
        elif isinstance(layer, layers.MaxPooling2D):
            layout.append({"type": "pool"})
        elif isinstance(layer, layers.Dense):
            k, b = layer.get_weights()
            arrays += [k.astype(np.float64), b.astype(np.float64)]
            layout.append({"type": "dense", "nin": k.shape[0], "nout": k.shape[1],
                           "activation": layer.activation.__name__})
    flat = np.concatenate([a.ravel() for a in arrays]).astype("<f4")
    return flat, layout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--check-dir", default=None,
                    help="where to write reference inputs and Keras outputs for checking the browser code")
    args = ap.parse_args()

    from src.models import build_small_cnn
    from src.train import build_file_lists, load_images, subsample
    from src import data as dataset_tools

    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {}
    keras_models = {}
    for key, tag in MODELS.items():
        m = build_small_cnn()
        m.load_weights(str(ROOT / "models" / f"{tag}.weights.h5"))
        keras_models[key] = m
        flat, layout = fold(m)
        (OUT / f"{key}.bin").write_bytes(flat.tobytes())
        manifest[key] = {"file": f"{key}.bin", "layout": layout, "floats": int(flat.size), "source": tag}
    (OUT / "model.json").write_text(json.dumps(manifest, indent=1))
    print("exported", {k: v["floats"] for k, v in manifest.items()})
    if not args.check_dir:
        return

    _, _, test_items, _ = build_file_lists(dataset_tools.resolve_root(args.root), dedup=True)
    items = subsample(test_items, args.n // 2)
    x = load_images([p for p, _ in items])
    check = {"n": len(x), "labels": [int(l) for _, l in items]}
    for key, m in keras_models.items():
        check[key] = [float(v) for v in m.predict(x.astype("float32"), verbose=0).ravel()]
    ref = Path(args.check_dir)
    ref.mkdir(parents=True, exist_ok=True)
    (ref / "inputs.bin").write_bytes(x.astype(np.uint8).tobytes())
    (ref / "expected.json").write_text(json.dumps(check))
    print("check images", len(x), "->", ref)


if __name__ == "__main__":
    main()
