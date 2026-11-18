"""
Local interface: upload an image, see what the trained detectors make of it.

Runs the Keras models with TensorFlow, on the same input the detectors were
trained on: the image is centre-cropped, downscaled to 32x32 and stored as JPEG
at quality 75, which is how CIFAKE stores both of its classes. Any of the audit
conditions can then be applied, so the effect of ordinary handling on each
detector's answer can be seen directly.

    python demo/app.py                  # opens http://127.0.0.1:7860
    python demo/app.py --models-dir models --share   # temporary public link
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import transforms as tf_ops  # noqa: E402
from src.models import BUILDERS  # noqa: E402

DETECTORS = {
    "Small CNN": ("small_cnn", "small_cnn"),
    "Small CNN, hardened": ("small_cnn", "small_cnn_hardened"),
    "EfficientNet-B0 (fine-tuned)": ("efficientnetb0", "efficientnetb0_finetuned"),
    "EfficientNet-B0 (fine-tuned), hardened": ("efficientnetb0", "efficientnetb0_finetuned_hardened"),
}
CONDITION_LABELS = {
    "clean": "none",
    "jpeg_q50": "JPEG, quality 50",
    "resize_0.50": "resize to half and back",
    "blur_1.0": "Gaussian blur, sigma 1",
    "social_q50": "resize to half + JPEG 50 (upload-like)",
    "sharpen": "sharpen (unsharp mask)",
}


def to_cifake_format(img: Image.Image) -> np.ndarray:
    """Centre crop, downscale to 32x32 and re-encode as JPEG 75, as in CIFAKE."""
    img = img.convert("RGB")
    s = min(img.size)
    left, top = (img.width - s) // 2, (img.height - s) // 2
    small = img.crop((left, top, left + s, top + s)).resize((32, 32), Image.Resampling.BICUBIC)
    buf = io.BytesIO()
    small.save(buf, format="JPEG", quality=75)
    return np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8)


def load_detectors(models_dir: Path) -> dict:
    loaded = {}
    for label, (arch, tag) in DETECTORS.items():
        path = models_dir / f"{tag}.weights.h5"
        if not path.exists():
            print(f"skipping {label}: {path} not found")
            continue
        model = BUILDERS[arch]()
        model.load_weights(str(path))
        loaded[label] = model
    if not loaded:
        raise SystemExit(f"no weights found in {models_dir}")
    return loaded


def predict(models: dict, img: Image.Image, condition: str) -> tuple[np.ndarray, dict]:
    x = to_cifake_format(img)
    if condition != "clean":
        x = tf_ops.CONDITIONS[condition](x)
    batch = x[None].astype("float32")
    scores = {label: float(m.predict(batch, verbose=0).ravel()[0]) for label, m in models.items()}
    return x, scores


def build_interface(models: dict):
    import gradio as gr

    def run(image, condition_label):
        if image is None:
            return None, "Upload an image."
        condition = next(k for k, v in CONDITION_LABELS.items() if v == condition_label)
        x, scores = predict(models, Image.fromarray(image), condition)
        rows = ["| Detector | P(generated) | Verdict |", "|---|---|---|"]
        for label, p in scores.items():
            rows.append(f"| {label} | {p:.3f} | {'generated' if p >= 0.5 else 'photograph'} |")
        preview = Image.fromarray(x).resize((256, 256), Image.Resampling.NEAREST)
        return preview, "\n".join(rows)

    with gr.Blocks(title="Real or generated?") as ui:
        gr.Markdown(
            "## Real or generated?\n"
            "Upload an image. It is brought to the format the detectors were trained on "
            "(centre crop, 32x32, JPEG 75); optionally apply one ordinary operation first. "
            "The detectors were trained on CIFAKE (Stable Diffusion 1.4) only: they are "
            "not expected to recognise other generators.")
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="numpy", label="Image")
                condition = gr.Dropdown(list(CONDITION_LABELS.values()), value="none",
                                        label="Operation applied before scoring")
                button = gr.Button("Classify", variant="primary")
            with gr.Column():
                preview = gr.Image(label="What the detectors see (32x32, enlarged)")
                table = gr.Markdown()
        button.click(run, [image, condition], [preview, table])
    return ui


def main() -> None:
    ap = argparse.ArgumentParser(description="Upload-and-predict interface for the trained detectors.")
    ap.add_argument("--models-dir", default=str(ROOT / "models"))
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="create a temporary public Gradio link")
    args = ap.parse_args()
    models = load_detectors(Path(args.models_dir))
    build_interface(models).launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
