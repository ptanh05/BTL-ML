# Lightweight YOLOv4 Helmet Detection

This branch contains a runnable PyTorch implementation of the six-model
ablation plan described in `Ke_Hoach_Trien_Khai_YOLOv4_Sensors2023_.docx`.
The target task is two-class detection on SHWD: `hat` and `person`.

## Project layout

```text
configs/       dataset and Model-1..6 YAML files
datasets/      Pascal-VOC/YOLO loader, transforms, split preparation
models/        CSPDarknet53, PP-LCNet, CoordAtt, PANet/PB, YOLO head
utils/         CIoU/SIoU, metrics, plots, checkpoint and Colab helpers
train.py       two-stage freeze/unfreeze training
eval.py        mAP, precision/recall/F1, FPS and benchmark CSV
detect.py      image/directory inference and visualisation
run_ablation.py sequential Model-1..6 training and comparison report
tests/         smoke tests for tensor interfaces and numerical stability
```

## Install

Use Python 3.12 and a CUDA-enabled PyTorch wheel when a GPU is available.
Then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

## Verify the implementation

```bash
python -m compileall -q models datasets utils train.py eval.py detect.py
pytest -q
python models/yolo.py
```

The model interface is fixed at 608x608 input and returns three raw YOLO
feature maps with shapes `(B, 21, 76, 76)`, `(B, 21, 38, 38)` and
`(B, 21, 19, 19)`.

## Prepare data

The checked-in split files point to images under `dataset/` and Pascal VOC
annotations under `dataset/annotations/`. To generate a deterministic split
from a fresh archive:

```bash
python -m datasets.prepare_data --zip path/to/SHWD.zip \
  --data-root dataset --split-dir datasets/splits --seed 42
```

## Train

For a smoke run:

```bash
python train.py --config configs/model6.yaml --data configs/dataset.yaml \
  --device cuda --dry-run --limit 50
```

For the two-stage experiment from the document:

```bash
python train.py --config configs/model6.yaml --data configs/dataset.yaml \
  --device cuda --output weights/model6
```

Training freezes the backbone for 50 epochs, then unfreezes all parameters
for 150 epochs, saves atomic checkpoints, and writes `history.json`.

## Evaluate and detect

```bash
python eval.py --weights weights/model6/model_6_best.pth \
  --config configs/model6.yaml --data configs/dataset.yaml --device cuda

python eval.py --benchmark-all --weights-dir weights --device cuda

python detect.py --weights weights/model6/model_6_best.pth \
  --config configs/model6.yaml --source dataset/test --output results/detections
```

Model-3/4 mAP values in the source paper contain an internal arithmetic
discrepancy; the implementation reports measured values rather than forcing
the paper's target numbers.

## Sequential ablation run

To train and evaluate every model in order, with a separate checkpoint and
history directory for each model:

```bash
python run_ablation.py --device cuda \
  --output-root weights/ablation \
  --results results/ablation
```

The runner writes `model_1.json` through `model_6.json`, one
`history.json` per model, `ablation_summary.csv`, and
`ablation_report.md`. Use `--dry-run --limit 8 --no-fps` first to validate the
whole sequence without starting a full 200-epoch experiment.
