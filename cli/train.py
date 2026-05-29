# cli/train.py
from __future__ import annotations

import argparse
from dataclasses import fields

import tensorflow as tf

from src.config import PipelineConfig
from src.data_loader import build_datasets
from src.evaluation import evaluate_model
from src.helpers import print_kv, print_section
from src.paths import ExperimentPaths
from src.reporting import create_reports
from src.training import train_model


def parse_args() -> argparse.Namespace:
    """Parse all CLI arguments for a training run."""
    parser = argparse.ArgumentParser(
        description="Chest X-ray training pipeline"
    )

    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--img_size", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--fine_tune_epochs", type=int, default=None)

    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--fine_tune_lr", type=float, default=None)

    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--unfreeze_last_n", type=int, default=None)

    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)

    parser.add_argument("--train_take", type=int, default=None)
    parser.add_argument("--val_take", type=int, default=None)
    parser.add_argument("--test_take", type=int, default=None)

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--val_split", type=float, default=None)

    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--mixed_precision", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--fine_tune", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use_class_weights", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--use_augmentation",
        action=argparse.BooleanOptionalAction,
        default=None,
    )

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> PipelineConfig:
    """Create config defaults from PipelineConfig and override only CLI-provided values."""
    config_fields = {field.name for field in fields(PipelineConfig)}

    overrides = {
        key: value
        for key, value in vars(args).items()
        if key in config_fields and value is not None
    }

    return PipelineConfig(**overrides)


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

    print_section("Run Initialization")
    print_kv("Run ID", paths.run_id)
    print_kv("Run Name", config.run_name)
    print_kv("Model", config.model_name)
    print_kv("Data Directory", config.data_dir)
    print_kv("Image Size", config.img_size)
    print_kv("Batch Size", config.batch_size)
    print_kv("Stage 1 Epochs", config.epochs)
    print_kv("Fine-Tuning Epochs", config.fine_tune_epochs if config.fine_tune else 0)
    print_kv("Learning Rate", config.learning_rate)
    print_kv("Fine-Tune LR", config.fine_tune_lr if config.fine_tune else "-")
    print_kv("Validation Split", config.val_split)
    print_kv("Seed", config.seed)
    print_kv("Data Augmentation", config.use_augmentation)
    print_kv("Class Weights", config.use_class_weights)
    print_kv("Mixed Precision", config.mixed_precision)

    print_section("Loading Data")
    data = build_datasets(config)

    print_kv("Classes", ", ".join(data.class_names))
    print_kv("Number of Classes", data.num_classes)

    print_section("Dataset Split")
    print_kv("Train Samples", data.train_samples)
    print_kv("Validation Samples", data.val_samples)
    print_kv("Test Samples", data.test_samples)
    print_kv("Validation Order", "deterministic (shuffle=False)")

    if data.class_counts is not None:
        print_section("Class Weights")
        for idx, class_name in enumerate(data.class_names):
            print_kv(f"Train {class_name}", int(data.class_counts[idx]))
        print_kv("Computed Weights", data.class_weights)

    print_section("Training")
    _, history_dict = train_model(
        config=config,
        data=data,
        paths=paths,
    )

    print_section("Evaluation on Test Set")
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

    print_section("Results Overview")
    print_kv("Best Model", paths.best_model_path.name)
    print_kv("Test Loss", f"{test_loss:.4f}")
    print_kv("Test Accuracy", f"{results.summary['results']['manual_test_accuracy']:.4f}")

    if "sparse_top_k_categorical_accuracy" in results.summary["results"]:
        print_kv(
            "Test Top-2 Accuracy",
            f"{results.summary['results']['sparse_top_k_categorical_accuracy']:.4f}",
        )

    print_kv("ROC-AUC OVR Macro", f"{results.summary['results']['roc_auc_ovr_macro']:.4f}")
    print_kv("ROC-AUC OVR Weighted", f"{results.summary['results']['roc_auc_ovr_weighted']:.4f}")
    print_kv("Macro Precision", f"{results.summary['results']['macro_precision']:.4f}")
    print_kv("Macro Recall", f"{results.summary['results']['macro_recall']:.4f}")
    print_kv("Macro F1", f"{results.summary['results']['macro_f1']:.4f}")
    print_kv("Weighted Precision", f"{results.summary['results']['weighted_precision']:.4f}")
    print_kv("Weighted Recall", f"{results.summary['results']['weighted_recall']:.4f}")
    print_kv("Weighted F1", f"{results.summary['results']['weighted_f1']:.4f}")
    print_kv("Model Path", paths.best_model_path)
    print_kv("Log Directory", paths.log_dir)
    print_kv("Metrics Directory", paths.met_dir)
    print_kv("Figures Directory", paths.fig_dir)

    print()
    print("Class-wise metrics:")
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