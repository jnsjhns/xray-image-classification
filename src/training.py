# src/training.py
from __future__ import annotations

from tensorflow import keras

from src.callbacks import create_stage1_callbacks, create_stage2_callbacks
from src.config import PipelineConfig
from src.data_loader import DatasetBundle
from src.helpers import merge_histories, save_json
from src.model import build_model, compile_model, unfreeze_layers
from src.paths import ExperimentPaths

# cli/train.py
from __future__ import annotations

import argparse

import tensorflow as tf

from src.config import PipelineConfig
from src.data_loader import build_datasets
from src.evaluation import evaluate_model
from src.helpers import print_kv, print_section
from src.paths import ExperimentPaths
from src.reporting import create_reports
from src.training import train_model


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for a training run."""
    parser = argparse.ArgumentParser(
        description="Image classification training pipeline"
    )

    parser.add_argument("--data_dir", type=str, default="data/XRAY")
    parser.add_argument("--img_size", type=int, default=299)
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fine_tune_epochs", type=int, default=10)

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--fine_tune_lr", type=float, default=1e-5)

    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--unfreeze_last_n", type=int, default=50)

    parser.add_argument("--run_name", type=str, default="xray_exp")
    parser.add_argument("--model_name", type=str, default="inceptionv3")

    parser.add_argument("--train_take", type=int, default=-1)
    parser.add_argument("--val_take", type=int, default=-1)
    parser.add_argument("--test_take", type=int, default=-1)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_split", type=float, default=0.1)

    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--fine_tune", action="store_true")
    parser.add_argument("--use_class_weights", action="store_true")
    parser.add_argument(
        "--use_augmentation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> PipelineConfig:
    """Create a PipelineConfig from parsed CLI arguments."""
    return PipelineConfig(
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        fine_tune_epochs=args.fine_tune_epochs,
        learning_rate=args.learning_rate,
        fine_tune_lr=args.fine_tune_lr,
        dropout=args.dropout,
        unfreeze_last_n=args.unfreeze_last_n,
        run_name=args.run_name,
        model_name=args.model_name,
        train_take=args.train_take,
        val_take=args.val_take,
        test_take=args.test_take,
        seed=args.seed,
        val_split=args.val_split,
        cache=args.cache,
        mixed_precision=args.mixed_precision,
        fine_tune=args.fine_tune,
        use_class_weights=args.use_class_weights,
        use_augmentation=args.use_augmentation,
    )


def main() -> None:
    args = parse_args()
    config = build_config(args)

    if not (0.0 < config.val_split < 1.0):
        raise ValueError("--val_split must be between 0 and 1.")

    tf.keras.utils.set_random_seed(config.seed)

    if config.mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    paths = ExperimentPaths.from_config(config)
    paths.create_directories()

    print_section("Run initialization")
    print_kv("Run ID", paths.run_id)
    print_kv("Run name", config.run_name)
    print_kv("Model", config.model_name)
    print_kv("Data directory", config.data_dir)
    print_kv("Image size", config.img_size)
    print_kv("Batch size", config.batch_size)
    print_kv("Stage 1 epochs", config.epochs)
    print_kv("Fine-tune epochs", config.fine_tune_epochs if config.fine_tune else 0)
    print_kv("Learning rate", config.learning_rate)
    print_kv("Fine-tune LR", config.fine_tune_lr if config.fine_tune else "-")
    print_kv("Validation split", config.val_split)
    print_kv("Seed", config.seed)
    print_kv("Data augmentation", config.use_augmentation)
    print_kv("Class weights", config.use_class_weights)
    print_kv("Mixed precision", config.mixed_precision)

    print_section("Loading data")
    data = build_datasets(config)

    print_kv("Classes", ", ".join(data.class_names))
    print_kv("Number of classes", data.num_classes)

    print_section("Dataset split")
    print_kv("Train samples", data.train_samples)
    print_kv("Validation samples", data.val_samples)
    print_kv("Test samples", data.test_samples)
    print_kv("Validation order", "deterministic (shuffle=False)")

    if data.class_counts is not None:
        print_section("Class weights")
        for idx, class_name in enumerate(data.class_names):
            print_kv(f"Train {class_name}", int(data.class_counts[idx]))
        print_kv("Computed weights", data.class_weights)

    print_section("Training")
    _, history_dict = train_model(
        config=config,
        data=data,
        paths=paths,
    )

    print_section("Evaluation on test set")
    results = evaluate_model(
        config=config,
        data=data,
        paths=paths,
    )

    create_reports(
        history_dict=history_dict,
        results=results,
        paths=paths,
    )

    test_loss = results.summary["results"].get("loss", float("nan"))
    if not isinstance(test_loss, (int, float)):
        test_loss = float("nan")

    print_section("Results overview")
    print_kv("Best model", paths.best_model_path.name)
    print_kv("Test loss", f"{test_loss:.4f}")
    print_kv("Test accuracy", f"{results.summary['results']['manual_test_accuracy']:.4f}")

    if "sparse_top_k_categorical_accuracy" in results.summary["results"]:
        print_kv(
            "Test top-2 accuracy",
            f"{results.summary['results']['sparse_top_k_categorical_accuracy']:.4f}",
        )

    print_kv("ROC-AUC OVR macro", f"{results.summary['results']['roc_auc_ovr_macro']:.4f}")
    print_kv("ROC-AUC OVR weighted", f"{results.summary['results']['roc_auc_ovr_weighted']:.4f}")
    print_kv("Macro precision", f"{results.summary['results']['macro_precision']:.4f}")
    print_kv("Macro recall", f"{results.summary['results']['macro_recall']:.4f}")
    print_kv("Macro F1", f"{results.summary['results']['macro_f1']:.4f}")
    print_kv("Weighted precision", f"{results.summary['results']['weighted_precision']:.4f}")
    print_kv("Weighted recall", f"{results.summary['results']['weighted_recall']:.4f}")
    print_kv("Weighted F1", f"{results.summary['results']['weighted_f1']:.4f}")
    print_kv("Model path", paths.best_model_path)
    print_kv("Log directory", paths.log_dir)
    print_kv("Metrics directory", paths.met_dir)
    print_kv("Figures directory", paths.fig_dir)

    print()
    print("Per-class metrics:")
    class_report_cols = ["precision", "recall", "f1-score", "support"]
    print(
        results.classification_report_df.loc[
            data.class_names,
            class_report_cols,
        ].round(4).to_string()
    )

    print()
    print(f"[INFO] Results saved to: {paths.output_root}")


if __name__ == "__main__":
    main()
def _build_config_payload(
    config: PipelineConfig,
    data: DatasetBundle,
) -> dict:
    """Build a serializable configuration and run metadata dictionary."""
    payload = config.to_dict()

    payload["class_names"] = data.class_names
    payload["num_classes"] = data.num_classes
    payload["train_samples_used"] = data.train_samples
    payload["val_samples_used"] = data.val_samples
    payload["test_samples_used"] = data.test_samples

    if data.class_counts is not None:
        payload["train_class_counts"] = {
            data.class_names[i]: int(data.class_counts[i])
            for i in range(data.num_classes)
        }

    if data.class_weights is not None:
        payload["class_weights"] = {
            str(k): float(v) for k, v in data.class_weights.items()
        }

    return payload


def train_model(
    config: PipelineConfig,
    data: DatasetBundle,
    paths: ExperimentPaths,
) -> tuple[keras.Model, dict]:
    """
    Run the full training pipeline:
    - Stage 1: feature extraction
    - optional Stage 2: fine-tuning
    - save training history and run configuration
    """
    model, base_model = build_model(config, data.num_classes)

    compile_model(
        model=model,
        learning_rate=config.learning_rate,
        num_classes=data.num_classes,
    )

    h1 = model.fit(
        data.train_ds,
        validation_data=data.val_ds,
        epochs=config.epochs,
        class_weight=data.class_weights,
        callbacks=create_stage1_callbacks(paths),
        verbose=1,
    )

    h2 = None
    if config.fine_tune:
        unfreeze_layers(base_model, config.unfreeze_last_n)

        compile_model(
            model=model,
            learning_rate=config.fine_tune_lr,
            num_classes=data.num_classes,
        )

        h2 = model.fit(
            data.train_ds,
            validation_data=data.val_ds,
            initial_epoch=config.epochs,
            epochs=config.epochs + config.fine_tune_epochs,
            class_weight=data.class_weights,
            callbacks=create_stage2_callbacks(paths),
            verbose=1,
        )

    history_dict = merge_histories(h1, h2)
    config_payload = _build_config_payload(config, data)

    save_json(paths.met_dir / "history.json", history_dict)
    save_json(paths.met_dir / "run_config.json", config_payload)

    return model, history_dict