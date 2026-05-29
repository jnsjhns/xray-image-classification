# X-Ray Image Classification

This project trains an image classification model for chest X-ray data using TensorFlow/Keras. The pipeline is modular and covers data loading, training, evaluation, and report generation.

## Project Structure

```text
xray-image-classification/
├── data/
│   └── chest_xray/
│       ├── train/
│       │   ├── NORMAL/
│       │   └── PNEUMONIA/
│       └── test/
│           ├── NORMAL/
│           └── PNEUMONIA/
├── cli/
│   └── train.py
│
├── src/
│   ├── callbacks.py
│   ├── config.py
│   ├── data_loader.py
│   ├── evaluation.py
│   ├── helpers.py
│   ├── model.py
│   ├── paths.py
│   ├── reporting.py
│   └── training.py
│
├── experiment_outputs/
│   └── <timestamp_run_name>/
│       ├── models/
│       ├── runs/
│       └── reports/
│           ├── figures/
│           └── metrics/
│
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.11
- Git
- Windows PowerShell, Command Prompt, or another terminal
- Optional: PyCharm or VS Code

## Clone the Repository

```bash
git clone <YOUR-REPOSITORY-URL>
cd xray-image-classification
```

## Create a Virtual Environment

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Windows Command Prompt

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Upgrade Pip

```bash
python -m pip install --upgrade pip
```

## Install all project dependencies

```bash
python -m pip install -r requirements.txt
```

## Prepare the Dataset

The dataset should follow a class-based directory structure that Keras can read directly:

```text
data/chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

## Run a Smoke Test

Use this minimal run to verify that the pipeline works end to end:

```bash
python -m cli.train \
  --train_take 1 \
  --val_take 1 \
  --test_take 1 \
  --epochs 1 \
  --fine_tune_epochs 0
  ```

## Run Training

A typical training command can override only the values that should differ from `src/config.py`:

```bash
python -m cli.train \
  --img_size 224 \
  --batch_size 16 \
  --epochs 5 \
  --fine_tune_epochs 3 \
  --learning_rate 1e-4 \
  --fine_tune_lr 1e-5 \
  --use_augmentation
```

The dataset path does not need to be passed if it matches the default in `src/config.py`:

```python
data_dir: Path = Path("data/chest_xray")
```
```

## Important CLI Arguments


```
All CLI arguments are optional overrides for the defaults in `src/config.py`.
```

| Argument | Description |
|---|---|
| `--data_dir` | Path to the dataset |
| `--img_size` | Target image size |
| `--batch_size` | Batch size |
| `--epochs` | Number of stage 1 epochs |
| `--fine_tune_epochs` | Number of fine-tuning epochs |
| `--learning_rate` | Learning rate for stage 1 |
| `--fine_tune_lr` | Learning rate for fine-tuning |
| `--run_name` | Name of the experiment |
| `--train_take`, `--val_take`, `--test_take` | Limit samples for smoke tests |
| `--use_augmentation` / `--no-use_augmentation` | Enable or disable augmentation |
| `--fine_tune` | Enable fine-tuning |
| `--use_class_weights` | Use class weights |
| `--mixed_precision` | Enable mixed precision |

## Outputs

Training outputs are typically stored in `experiment_outputs/`, including saved models, metrics, plots, and reports for each run.

Generated training artifacts usually should not be committed to Git.

Recommended `.gitignore` entries:

```gitignore
.venv/
__pycache__/
*.pyc
experiment_outputs/
*.keras
*.h5
.DS_Store
```

## Troubleshooting

### TensorFlow is not found

Check whether TensorFlow is installed in the active environment:

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

If this fails, the wrong environment is usually active or TensorFlow was not installed in the current `.venv`.

### ModuleNotFoundError: No module named 'src'

Run the training command from the project root using module mode:

```bash
python -m cli.train
```

### Old or broken environments

If package conflicts appear, creating a fresh `venv` is often faster and cleaner than repairing the global Python installation.

## Development Workflow

For new features or dataset changes, this workflow is recommended:

1. Create a new branch.
2. Activate `.venv`.
3. Implement your changes.
4. Run a smoke test.
5. Start a longer training run only after the smoke test passes.

## Git Notes

- Commit source code, configuration, and documentation.
- Do not commit generated models, logs, or complete experiment outputs.
- Only commit small example metrics if they are intentionally included as reference artifacts.
