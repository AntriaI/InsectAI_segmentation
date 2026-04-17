# Insect Binary Classification

This project implements a **binary image classification pipeline**

---

## Project Structure

```
src/
 ├── datafeeder/        # Dataset + DataLoader
 ├── loss/              # Loss functions (BCE, Focal)
 ├── models/            # Model definitions (EfficientNet)
 ├── train/             # Training loop
 ├── utils/             # Logger, config loader
experiments/
 ├── exp_01/
 │    ├── config.json   # Experiment configuration
 │    ├── run.py        # Entry point
 │    ├── logs/         # Training logs
 │    ├── tensorboard/  # TensorBoard logs
 │    ├── models/       # Saved models
```

---

## Dataset Format

Organize your dataset like this:

```
dataset/
 ├── insect/
 │    ├── img1.jpg
 │    ├── img2.jpg
 ├── no-insect/
      ├── img3.jpg
      ├── img4.jpg
```

---

## Configuration

Edit `config.json`:

```json
{
  "data_path": "../data/samp_data",
  "batch_size": 16,
  "seed_flag": true,
  "training": {
    "loss": "BCEWithLogitsLoss",
    "lr": 0.0001,
    "epochs": 100,
    "save_every": 25
  }
}
```

Supported losses:

* `"BCEWithLogitsLoss"`
* `"FocalLoss"`

---

## Run Training

```bash
python -m experiments.exp_01.run
```

---

## TensorBoard

```bash
tensorboard --logdir=experiments/exp_01/tensorboard
```

---

## Output

* Logs → `experiments/exp_01/logs`
* Models → `experiments/exp_01/models`
* TensorBoard → training curves + metrics

---