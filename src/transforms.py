"""
Image-degradation operations.

These are the ordinary things that happen to an image between being created and
being seen: it gets re-encoded as JPEG, resized to fit a layout, slightly
blurred, and picks up noise. The same operations are used in two places, and the
distinction matters:

  * build_degraded_set()  - applied once to a held-out test set, producing fixed
    evaluation conditions. Deterministic, so every model is scored on byte-identical
    inputs.
  * tf_training_degradation() - applied randomly to training images inside the
    input pipeline, so a model learns features that survive these operations
    rather than features that depend on a pristine encode. It draws parameters
    from continuous ranges and varies the resampling kernel, so the fixed
    evaluation conditions are points inside the training distribution rather
    than copies of it.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageFilter

# ---------------------------------------------------------------- numpy / PIL


def jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    """Re-encode through JPEG at the given quality factor."""
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    with Image.open(buf) as out:
        return np.array(out.convert("RGB"))


def resize_roundtrip(img: np.ndarray, factor: float = 0.5) -> np.ndarray:
    """Downscale then restore the original size, as a layout resize would."""
    h, w = img.shape[:2]
    small = (max(1, int(w * factor)), max(1, int(h * factor)))
    with Image.fromarray(img) as im:
        return np.array(im.resize(small, Image.BICUBIC).resize((w, h), Image.BICUBIC))


def blur(img: np.ndarray, sigma: float) -> np.ndarray:
    with Image.fromarray(img) as im:
        return np.array(im.filter(ImageFilter.GaussianBlur(radius=float(sigma))))


def gaussian_noise(img: np.ndarray, sigma: float, seed: int = 0) -> np.ndarray:
    """Additive Gaussian noise, rounded to the nearest grey level."""
    rng = np.random.default_rng(seed)
    noisy = img.astype(np.float32) + rng.normal(0.0, sigma, img.shape)
    return np.clip(np.rint(noisy), 0, 255).astype(np.uint8)


def webp(img: np.ndarray, quality: int) -> np.ndarray:
    """Re-encode through WebP, a codec absent from the training-time degradations."""
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="WEBP", quality=int(quality))
    buf.seek(0)
    with Image.open(buf) as out:
        return np.array(out.convert("RGB"))


def resize_roundtrip_lanczos(img: np.ndarray, factor: float = 0.5) -> np.ndarray:
    """Resize round trip with a Lanczos kernel, which training never uses."""
    h, w = img.shape[:2]
    small = (max(1, int(w * factor)), max(1, int(h * factor)))
    with Image.fromarray(img) as im:
        return np.array(im.resize(small, Image.LANCZOS).resize((w, h), Image.LANCZOS))


def sharpen(img: np.ndarray, percent: int = 150) -> np.ndarray:
    """Unsharp masking, as applied by many upload pipelines. It amplifies the
    high-frequency band rather than suppressing it."""
    with Image.fromarray(img) as im:
        return np.array(im.filter(ImageFilter.UnsharpMask(radius=1, percent=percent, threshold=0)))


def social_pipeline(img: np.ndarray, quality: int = 70, factor: float = 0.5) -> np.ndarray:
    """Resize then re-encode - the usual fate of an image posted online."""
    return jpeg(resize_roundtrip(img, factor), quality)


#: Evaluation grid. Each entry maps a condition name to a callable.
CONDITIONS = {
    "clean": lambda x: x,
    "jpeg_q90": lambda x: jpeg(x, 90),
    "jpeg_q70": lambda x: jpeg(x, 70),
    "jpeg_q50": lambda x: jpeg(x, 50),
    "jpeg_q30": lambda x: jpeg(x, 30),
    "resize_0.50": lambda x: resize_roundtrip(x, 0.50),
    "resize_0.25": lambda x: resize_roundtrip(x, 0.25),
    "blur_0.5": lambda x: blur(x, 0.5),
    "blur_1.0": lambda x: blur(x, 1.0),
    "noise_5": lambda x: gaussian_noise(x, 5.0),
    "noise_10": lambda x: gaussian_noise(x, 10.0),
    "social_q70": lambda x: social_pipeline(x, 70, 0.5),
    "social_q50": lambda x: social_pipeline(x, 50, 0.5),
    # Held out: none of these operations, codecs or kernels appear in the
    # training-time degradation, so they test generalisation beyond it.
    "webp_q50": lambda x: webp(x, 50),
    "resize_0.50_lanczos": lambda x: resize_roundtrip_lanczos(x, 0.50),
    "sharpen": lambda x: sharpen(x, 150),
}

#: Conditions whose operations are outside the training-time degradation family.
HELD_OUT = ("webp_q50", "resize_0.50_lanczos", "sharpen")

#: Noise conditions draw an independent, repeatable field for every image position.
NOISE_SIGMA = {"noise_5": 5.0, "noise_10": 10.0}


def build_degraded_set(images: np.ndarray, condition: str) -> np.ndarray:
    """Apply one named condition to a stack of uint8 HWC images."""
    if condition not in CONDITIONS:
        raise KeyError(f"unknown condition {condition!r}; expected one of {sorted(CONDITIONS)}")
    if condition in NOISE_SIGMA:
        sigma = NOISE_SIGMA[condition]
        return np.stack([gaussian_noise(im, sigma, seed=i) for i, im in enumerate(images)])
    fn = CONDITIONS[condition]
    return np.stack([fn(im) for im in images])


# ----------------------------------------------------------------- tensorflow


def tf_training_degradation(p_clean: float = 0.2):
    """Return a tf.data map function that randomly degrades one training image.

    Input and output are float32 images in [0, 255]. With probability `p_clean`
    the image passes through untouched, so pristine statistics stay in the
    training distribution. Otherwise each stage below is applied independently,
    in the order an image typically meets them before being published:

      blur    p=0.3   Gaussian, sigma 0.3-1.5
      resize  p=0.5   round trip at scale 0.25-1.0; downscale kernel drawn from
                      bilinear / bicubic / area, upscale from bilinear / bicubic
      noise   p=0.3   Gaussian, sigma 0-10
      JPEG    p=0.6   quality 30-95

    The work runs on the CPU in parallel with training rather than inside the
    model graph, so it costs no accelerator time.
    """
    import tensorflow as tf

    radius = 3
    taps = tf.range(-radius, radius + 1, dtype=tf.float32)

    def gaussian_blur(x, sigma):
        k = tf.exp(-0.5 * tf.square(taps / sigma))
        k = k / tf.reduce_sum(k)
        kx = tf.tile(tf.reshape(k, [1, -1, 1, 1]), [1, 1, 3, 1])
        ky = tf.tile(tf.reshape(k, [-1, 1, 1, 1]), [1, 1, 3, 1])
        y = tf.pad(x[None], [[0, 0], [radius, radius], [radius, radius], [0, 0]], mode="REFLECT")
        y = tf.nn.depthwise_conv2d(y, kx, [1, 1, 1, 1], "VALID")
        y = tf.nn.depthwise_conv2d(y, ky, [1, 1, 1, 1], "VALID")
        return y[0]

    def resize_roundtrip_tf(x, scale):
        size = tf.shape(x)[:2]
        small = tf.maximum(1, tf.cast(tf.round(tf.cast(size, tf.float32) * scale), tf.int32))
        down = tf.random.uniform([], 0, 3, dtype=tf.int32)
        up = tf.random.uniform([], 0, 2, dtype=tf.int32)
        y = tf.switch_case(down, [
            lambda: tf.image.resize(x, small, method="bilinear", antialias=True),
            lambda: tf.image.resize(x, small, method="bicubic", antialias=True),
            lambda: tf.image.resize(x, small, method="area"),
        ])
        return tf.switch_case(up, [
            lambda: tf.image.resize(y, size, method="bilinear"),
            lambda: tf.image.resize(y, size, method="bicubic"),
        ])

    def chance(p):
        return tf.random.uniform([]) < p

    def degrade(x):
        x = tf.cond(chance(0.3),
                    lambda: gaussian_blur(x, tf.random.uniform([], 0.3, 1.5)), lambda: x)
        x = tf.cond(chance(0.5),
                    lambda: resize_roundtrip_tf(x, tf.random.uniform([], 0.25, 1.0)), lambda: x)
        x = tf.cond(chance(0.3),
                    lambda: x + tf.random.normal(tf.shape(x), 0.0, tf.random.uniform([], 0.0, 10.0)),
                    lambda: x)
        x = tf.clip_by_value(x, 0.0, 255.0)

        def jpeg_tf():
            u8 = tf.cast(tf.round(x), tf.uint8)
            q = tf.random.uniform([], 30, 96, dtype=tf.int32)
            return tf.cast(tf.image.adjust_jpeg_quality(u8, q), tf.float32)

        x = tf.cond(chance(0.6), jpeg_tf, lambda: x)
        x.set_shape([None, None, 3])
        return x

    def apply(image, label):
        image = tf.cast(image, tf.float32)
        out = tf.cond(chance(p_clean), lambda: image, lambda: degrade(image))
        out.set_shape(image.shape)
        return out, label

    return apply
