# cifake-robustness

How much of a CIFAKE detector's accuracy survives ordinary image handling?

Detectors that tell photographs from AI-generated images are usually scored on
pristine benchmark images. This repository trains five CNN detectors on
[CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
(32×32 images; generated class from Stable Diffusion 1.4), re-scores them after
the operations an image goes through before anyone sees it (JPEG and WebP
re-encoding, resizing, blur, noise, sharpening), explains the result in the
frequency domain, and tests whether training on degraded images closes the gap.

**Scope.** The detectors are trained and tested on CIFAKE. They are not
general-purpose detectors of AI-generated images: on images from other
generators they are close to chance.

- **Notebook:** [`colab.ipynb`](colab.ipynb) runs on Google Colab: it downloads
  CIFAKE and the trained weights, re-scores the detectors and retrains one.
- **Weights:** [huggingface.co/thisisdawit/cifake-robustness](https://huggingface.co/thisisdawit/cifake-robustness)
- **Demo:** [huggingface.co/spaces/thisisdawit/cifake-robustness](https://huggingface.co/spaces/thisisdawit/cifake-robustness),
  in the browser; apply a resize, blur, JPEG or sharpening and watch each detector's verdict.

## Findings

| | clean accuracy | after resize ½ + JPEG 50 | generated images still detected |
|---|---|---|---|
| Small CNN | 0.959 | 0.646 | 31% |
| ResNet-50 (frozen) | 0.924 | 0.796 | 62% |
| EfficientNet-B0 (frozen) | 0.880 | 0.790 | 65% |
| ResNet-50 (fine-tuned) | 0.982 | 0.599 | 20% |
| EfficientNet-B0 (fine-tuned) | 0.985 | 0.600 | 20% |
| Small CNN, **hardened** | 0.962 | 0.928 | 93% |
| EfficientNet-B0 (fine-tuned), **hardened** | 0.979 | 0.954 | 96% |

Full deduplicated CIFAKE test set (19,962 images). Source:
[`results/audit_*.json`](results/).

- **Resizing, not compression.** JPEG at quality 50 costs 1.4–3.5 accuracy
  points. A half-scale resize round trip costs up to 38.
- **One-sided and confident.** Degraded photographs are still recognised;
  generated images pass as real. The fine-tuned EfficientNet-B0 is right on
  60% of resized-and-compressed images while reporting 96% confidence.
- **The spectrum points to the cause.** Generated CIFAKE images carry excess
  high-frequency energy. Resizing and blurring push it below the level of an
  untouched photograph; sharpening pushes photographs the other way, and the
  detectors then call real images generated.
- **The benchmark ranking reverses.** Frozen ImageNet backbones, the weakest
  detectors on clean images, are the most robust and stay calibrated.
- **Training on degraded images fixes most of it.** Hardened detectors keep
  their clean accuracy, recover 30+ points under resizing (three seeds for the Small CNN),
  stay calibrated, and handle a resampling kernel never seen in training.
  Sharpening still makes them call about a third of photographs generated.
- **No transfer beyond Stable Diffusion.** On eight GenImage generators the
  detectors are near chance outside the Stable Diffusion family.

## Layout

```
src/
  data.py         dataset access, image categories, duplicate check
  transforms.py   test-time degradations and the training-time random degradation
  models.py       Small CNN, ResNet-50 and EfficientNet-B0 detectors
  train.py        training on the deduplicated split, optional hardening
  evaluate.py     robustness audit: accuracy, per-class recall, AUC, calibration
  analysis.py     spectra, per-category accuracy, Grad-CAM, GenImage, ROC curves
demo/             browser demo (web/) and a local TensorFlow interface (app.py)
results/          machine-generated metrics for every reported run
colab.ipynb       end-to-end run on Google Colab
```

## Running

```bash
python -m pip install -r requirements.txt
kaggle datasets download -d birdy654/cifake-real-and-ai-generated-synthetic-images --unzip -p ./data/cifake
```

Train a detector (the settings of the reported runs):

```bash
python -m src.train --root ./data/cifake --dedup --model small_cnn --epochs 20
python -m src.train --root ./data/cifake --dedup --model small_cnn --epochs 30 --harden
python -m src.train --root ./data/cifake --dedup --model resnet50 --epochs 20
python -m src.train --root ./data/cifake --dedup --model resnet50 --epochs 12 --lr 1e-4 --finetune
python -m src.train --root ./data/cifake --dedup --model efficientnetb0 --epochs 20
python -m src.train --root ./data/cifake --dedup --model efficientnetb0 --epochs 12 --lr 1e-4 --finetune
python -m src.train --root ./data/cifake --dedup --model efficientnetb0 --epochs 16 --lr 1e-4 --finetune --harden
```

Batch size is 128 and the seed 1337 throughout; additional seeds use `--seed 1` and
`--seed 2`. Audit a trained detector under all sixteen conditions, and run the analyses:

```bash
python -m src.evaluate --root ./data/cifake --weights models/small_cnn.weights.h5 --model small_cnn
python -m src.analysis spectrum --root ./data/cifake --audit results/audit_small_cnn.json
python -m src.analysis explain --root ./data/cifake
```

Re-scoring the released weights reproduces `results/` to within 0.001. Retraining
gives a new model that agrees within run-to-run variation (about one point of
clean accuracy), not exactly. Training a backbone needs a GPU.

The deduplicated split removes 378 images that appear in both halves of the
published split and 958 within-split duplicates. Results on the published split
are kept for comparison (`*_published_split`).

## Licence

Code released under the MIT Licence. The CIFAKE and GenImage datasets carry
their own terms; see the dataset pages.
