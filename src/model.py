# src/model.py
from __future__ import annotations

from collections.abc import Callable

from tensorflow import keras
from tensorflow.keras import layers

from src.config import PipelineConfig


def get_metrics(num_classes: int) -> list[keras.metrics.Metric]:
    """Create the default training metrics."""
    metrics: list[keras.metrics.Metric] = [
        keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
    ]

    if num_classes > 2:
        metrics.append(
            keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top2_acc")
        )

    return metrics

"""
def build_augmentation_layer(config: PipelineConfig) -> keras.Sequential:
    
    #Create an optional data augmentation pipeline.
    #Note:
    #Conservative augmentation for chest X-rays: small rotation and zoom only.
    #Horizontal flipping is often avoided because anatomical orientation matters.

    return keras.Sequential(
        [
            layers.RandomRotation(0.05, seed=config.seed),
            layers.RandomZoom(0.10, seed=config.seed),
        ],
        name="data_augmentation",
    )
"""


def build_augmentation_layer(config: PipelineConfig) -> keras.Sequential | None:
    """
    Create a configurable augmentation pipeline.
    Individual augmentation components can be enabled separately.
    """

    if not config.use_augmentation:
        return None

    aug_layers = []

    if config.aug_rotation:
        aug_layers.append(
            layers.RandomRotation(
                config.aug_rotation_factor,
                seed=config.seed,
            )
        )

    if config.aug_zoom:
        aug_layers.append(
            layers.RandomZoom(
                config.aug_zoom_factor,
                seed=config.seed,
            )
        )

    if config.aug_contrast:
        aug_layers.append(
            layers.RandomContrast(
                config.aug_contrast_factor,
                seed=config.seed,
            )
        )

    if not aug_layers:
        return None

    return keras.Sequential(
        aug_layers,
        name="data_augmentation",
    )

def get_preprocess_fn(model_name: str) -> Callable:
    """
    Return the preprocessing function that matches the selected architecture.

    Important:
    Each Keras application expects its own preprocessing function.
    """
    model_name = model_name.lower()

    if model_name == "inceptionv3":
        return keras.applications.inception_v3.preprocess_input

    if model_name == "cnn_scratch":
        return lambda x: x

    if model_name == "resnet50":
        return keras.applications.resnet50.preprocess_input

    if model_name == "efficientnetb0":
        return keras.applications.efficientnet.preprocess_input

    if model_name == "efficientnetb3":
        return keras.applications.efficientnet.preprocess_input

    raise ValueError(f"Unsupported model: {model_name}")


def build_backbone(config: PipelineConfig) -> keras.Model:
    """
    Build the pretrained backbone without the classification head.
    """
    model_name = config.model_name.lower()

    if model_name == "inceptionv3":
        base_model = keras.applications.InceptionV3(
            weights="imagenet",
            include_top=False,
            input_shape=(config.img_size, config.img_size, 3),
        )
        base_model.trainable = False
        return base_model


    if model_name == "resnet50":
        base_model = keras.applications.ResNet50(
            weights="imagenet",
            include_top=False,
            input_shape=(config.img_size, config.img_size, 3),
        )
        base_model.trainable = False
        return base_model

    if model_name == "efficientnetb0":
        base_model = keras.applications.EfficientNetB0(
            weights="imagenet",
            include_top=False,
            input_shape=(config.img_size, config.img_size, 3),
        )
        base_model.trainable = False
        return base_model

    if model_name == "efficientnetb3":
        base_model = keras.applications.EfficientNetB3(
            weights="imagenet",
            include_top=False,
            input_shape=(config.img_size, config.img_size, 3),
        )
        base_model.trainable = False
        return base_model

    raise ValueError(f"Unsupported model: {model_name}")


def build_cnn_from_scratch(
    config: PipelineConfig,
    num_classes: int,
) -> keras.Model:

    inputs = keras.Input(
        shape=(config.img_size, config.img_size, 3),
        name="input_image",
    )

    x = layers.Rescaling(1.0 / 255)(inputs)

    augmentation_layer = build_augmentation_layer(config)

    if augmentation_layer is not None:
        x = augmentation_layer(x)

    x = layers.Conv2D(
        32,
        3,
        activation="relu",
        padding="same",
    )(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(
        64,
        3,
        activation="relu",
        padding="same",
    )(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(
        128,
        3,
        activation="relu",
        padding="same",
    )(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(
        256,
        3,
        activation="relu",
        padding="same",
    )(x)

    # Grad-CAM
    gradcam_features = layers.Identity(
        name="gradcam_features"
    )(x)

    x = layers.GlobalAveragePooling2D(name="global_avg_pooling")(gradcam_features)

    x = layers.Dense(256, activation="relu", name="scratch_dense_1")(x)
    x = layers.Dropout(config.dropout, seed=config.seed, name="head_dropout_1")(x)

    x = layers.Dense(128, activation="relu", name="scratch_dense_2")(x)
    x = layers.Dropout(config.dropout, seed=config.seed, name="head_dropout_2")(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        dtype="float32",
        name="predictions",
    )(x)

    return keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="xray_cnn_scratch",
    )


def build_model(
    config: PipelineConfig,
    num_classes: int,
) -> tuple[keras.Model, keras.Model]:
    """
        Build the full model from augmentation, preprocessing, backbone, and head.
        Returns both the full model and the backbone model.
    """

    if config.model_name.lower() == "cnn_scratch":
        model = build_cnn_from_scratch(
            config=config,
            num_classes=num_classes,
        )
        return model, model

    # InceptionV3 / transfer learning
    inputs = keras.Input(
        shape=(config.img_size, config.img_size, 3),
        name="input_image",
    )

    # Optional data augmentation
    augmentation_layer = build_augmentation_layer(config)

    if augmentation_layer is not None:
        x = augmentation_layer(inputs)
    else:
        x = inputs

    # Preprocessing (e.g. InceptionV3 preprocess_input)
    preprocess_fn = get_preprocess_fn(config.model_name)
    x = preprocess_fn(x)

    # Backbone (e.g. InceptionV3 without top)
    base_model = build_backbone(config)
    x = base_model(x, training=False)

    # Grad-CAM anchor: last convolutional feature maps
    gradcam_features = layers.Identity(name="gradcam_features")(x)
    # gradcam_features with Shape (H, W, C)

    # Classification head
    x = layers.GlobalAveragePooling2D(name="global_avg_pooling")(gradcam_features)
    x = layers.Dropout(config.dropout, seed=config.seed, name="head_dropout")(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        dtype="float32",
        name="predictions",
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name=f"xray_{config.model_name.lower()}",
    )

    return model, base_model


def compile_model(
    model: keras.Model,
    learning_rate: float,
    num_classes: int,
) -> None:
    """Compile the model with optimizer, loss, and metrics."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=get_metrics(num_classes),
    )


def unfreeze_layers(base_model: keras.Model, unfreeze_last_n: int) -> None:
    """
    Enable fine-tuning for the last N layers of the backbone.

    Batch normalization layers remain frozen.
    """
    base_model.trainable = True
    split_idx = max(0, len(base_model.layers) - unfreeze_last_n)

    for i, layer in enumerate(base_model.layers):
        if i < split_idx or isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

