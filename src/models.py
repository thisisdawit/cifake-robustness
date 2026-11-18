"""
Model architectures.

Three detectors are compared. The first is written from scratch and sized for
32x32 input; it is the reference point against which the pretrained backbones
are judged. The backbones expect far larger inputs than CIFAKE provides, so the
input is upsampled inside the model rather than on disk - the resolution
mismatch is itself one of the things being measured.
"""
from __future__ import annotations


def build_small_cnn(input_shape=(32, 32, 3), dropout=0.3):
    from tensorflow import keras
    from tensorflow.keras import layers


    inp = keras.Input(shape=input_shape, name="image")
    x = layers.Rescaling(1.0 / 255)(inp)
    x = layers.RandomFlip("horizontal")(x)

    for filters in (32, 64, 128):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="sigmoid", name="p_fake")(x)
    return keras.Model(inp, out, name="small_cnn")


def build_transfer(backbone="resnet50", input_shape=(32, 32, 3), upsample_to=96,
                   trainable_backbone=False, dropout=0.3):
    from tensorflow import keras
    from tensorflow.keras import layers


    backbones = {
        "resnet50": (keras.applications.ResNet50, keras.applications.resnet50.preprocess_input),
        "efficientnetb0": (keras.applications.EfficientNetB0, lambda z: z),
    }
    if backbone not in backbones:
        raise KeyError(f"unknown backbone {backbone!r}; expected one of {sorted(backbones)}")
    ctor, preprocess = backbones[backbone]

    inp = keras.Input(shape=input_shape, name="image")
    x = layers.Rescaling(1.0 / 255)(inp)
    x = layers.RandomFlip("horizontal")(x)
    # Bilinear rather than bicubic: ResizeBicubic has no XLA GPU kernel, and
    # graphs are XLA-compiled on multi-GPU runtimes. The difference is
    # negligible when upsampling 32px inputs.
    x = layers.Resizing(upsample_to, upsample_to, interpolation="bilinear")(x)
    x = layers.Rescaling(255.0)(x)
    x = layers.Lambda(preprocess, name="backbone_preprocess")(x)

    base = ctor(include_top=False, weights="imagenet",
                input_shape=(upsample_to, upsample_to, 3))
    base.trainable = trainable_backbone
    x = base(x, training=False)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="sigmoid", name="p_fake")(x)
    return keras.Model(inp, out, name=f"{backbone}_t{upsample_to}")


BUILDERS = {
    "small_cnn": build_small_cnn,
    "resnet50": lambda **kw: build_transfer(backbone="resnet50", **kw),
    "efficientnetb0": lambda **kw: build_transfer(backbone="efficientnetb0", **kw),
}
