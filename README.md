# Lightweight YOLOv4 Helmet Detection

This branch implements the six-model ablation in Chen et al.,
*Lightweight Helmet Detection Algorithm Using an Improved YOLOv4*, Sensors 2023,
23, 1256 (`sensors-23-01256-v2.pdf`, DOI 10.3390/s23031256).
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
  --device cuda --output weights/model6 --backbone-weights path/to/backbone.pth
```

The default `paper` protocol requires compatible pretrained backbone weights
and the cleaned SHWD counts from Section 4.1 (7004 images, 7709 hat, 101174 person).
It verifies the Section 4.2 schedule and refuses smoke/development checkpoints.
Training freezes the backbone (including
BatchNorm statistics) for 50 epochs, then unfreezes all parameters for 150 epochs.
For development on the dataset currently in this project, use `--from-scratch`:

```bash
python train.py --config configs/model6.yaml --from-scratch --device cuda --output weights/development_model6
python -m datasets.audit --data configs/dataset.yaml
```

This trains all parameters from epoch 1 using `lr_unfreeze` and
`batch_size_unfreeze`, and records `protocol: development`; it is not a replication
of the paper's transfer-learning experiment. `--dry-run` records `protocol: smoke`.
Pretrained files must match this PyTorch backbone's state
dictionary; raw Darknet or Paddle weights require conversion first.

Atomic checkpoints save the optimizer, scheduler, AMP scaler, training phase,
positive `best_val_loss`, and history. Resume defaults to saving in the checkpoint's
directory unless `--output` is supplied:

```bash
python train.py --config configs/model6.yaml --resume weights/model6/last.pth
```

Full paper experiments default to FP32; precision is not specified by the paper.
Optional CUDA AMP uses BF16 on supported GPUs for its wider numerical range, falling back
to FP16 with dynamic gradient scaling on older GPUs. Box geometry stays in FP32.

Model-1/2 now use the full CSPDarknet53 backbone (CSP depths 1/2/8/8/4 and Mish),
and PANet uses alternating 1x1/3x3 convolutions with LeakyReLU. The `paper` PP-LCNet
layout follows Figures 1/2 literally: stem 16, groups 64x2, 128x2, 256x2, 512x7;
3x3 DW in the first groups and 5x5 DW in the last group. A separate `canonical`
layout is available for explicitly chosen development experiments; it is not the
diagram layout. Its weights cannot be loaded into the paper layout unchanged.

PANet retains output widths (128, 256, 512) across the backbone ablation. Its first
and lateral convolutions accept the actual backbone widths directly, with no
extra input projections. PB follows Figure 4: fine-to-coarse fusion, then
coarse-to-fine fusion with the middle-layer residual. It retains per-level widths
instead of collapsing all three head inputs to 128 channels. SIoU uses
`siou_theta` and `siou_eps` from YAML.
Box/class losses are averaged over positive targets to preserve their weighting
when batch size changes. Target assignment uses width/height IoU over all nine
anchors (best match plus matches above threshold), resolves cell collisions, and
ignores overlapping unassigned predictions in background objectness loss.

### Details not specified by the paper

The last-two SE policy, CA reduction 32, retained internal neck/head widths,
channel-changing fusion convolutions, SIoU theta 4, unfreeze LR .0001, and loss
gains are explicit implementation choices. Figures 1/2 show SE after PW, but do
not state exactly which blocks use it. `pplcnet_se_blocks` makes that choice visible.
Global-anchor IoU .213 and ignore IoU .7 follow YOLOv4 conventions, not recovered
author settings. The paper also omits the exact split, seed, augmentation recipe,
AP IoU/confidence thresholds, serialization format for model size, and FPS timing
scope. Current FPS measures model forward only; AP uses IoU .5 and the selected
confidence cutoff. The implementation does not force the Table 1 size or mAP.
In Eq. (3), the paper's `c_w`/`c_h` notation is ambiguous with the center offsets
used in the angle diagram; distance normalization uses the enclosing rectangle
dimensions, as in conventional SIoU implementations, rather than dividing a
center offset by itself.

For the literal figure layout with these stated choices, measured FP32 parameter
sizes are 243.923, 136.117, 41.667, 41.803, 44.698, and 44.698 MiB for Model-1..6.
They do not match every Table 1 size. This is a reconstruction of the published
diagrams, not a verified copy of the authors' unpublished implementation.

The project contains the original 7581-image SHWD (9044 hat, 111514 person), and
only short smoke/timing checkpoints were found. No verified cleaned 7004-image
release or separate compatible pretrained backbone was found locally. Matching
the aggregate counts alone does not verify that the labels are the authors' labels;
the audit records this limitation. Do not delete arbitrary examples to match counts.

Checkpoints from the previous compact architectures are incompatible with the
updated models. Start a new experiment directory; evaluation and resume require
matching weights and fail on incompatible state dictionaries. Existing result
files describe the old implementation and should not be used as new benchmarks.

The Colab notebook handles archives containing a top-level `dataset/` directory,
generates `configs/dataset_colab.yaml` with an absolute SSD path, and uses it for
training, resume, evaluation and detection.

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
python run_ablation.py --device cuda --backbone-weights-dir path/to/pretrained \
  --output-root weights/ablation \
  --results results/ablation
```

The runner writes `model_1.json` through `model_6.json`, one
`history.json` per model, `ablation_summary.csv`, and
`ablation_report.md`. Use `--dry-run --limit 8 --no-fps` first to validate the
whole sequence without starting a full 200-epoch experiment.
The pretrained directory contains `cspdarknet53.pth` and `pplcnet_paper.pth`, or
each YAML can set `training.backbone_weights` separately. Use `--from-scratch`
for a development ablation on the current dataset.
