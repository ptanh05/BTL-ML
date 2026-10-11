from pathlib import Path

import torch

from datasets.shwd import SHWDDataset
from models.attention import CoordAtt
from models.pp_lcnet import PPLCNet
from models.yolo import YOLOv4Variant, MODEL_CFGS
from utils.loss import siou_loss
from utils.metrics import calculate_map


def test_pplcnet_interface():
    model = PPLCNet().eval()
    with torch.no_grad():
        outputs = model(torch.randn(1, 3, 608, 608))
    assert [tuple(output.shape) for output in outputs] == [
        (1, 128, 76, 76), (1, 256, 38, 38), (1, 512, 19, 19)
    ]


def test_coordinate_attention_preserves_shape():
    x = torch.randn(2, 128, 76, 76)
    assert CoordAtt(128)(x).shape == x.shape


def test_model6_head_interface():
    model = YOLOv4Variant(MODEL_CFGS[6]).eval()
    with torch.no_grad():
        outputs = model(torch.randn(1, 3, 608, 608))
    assert [tuple(output.shape) for output in outputs] == [
        (1, 21, 76, 76), (1, 21, 38, 38), (1, 21, 19, 19)
    ]


def test_siou_is_finite_at_identical_centres():
    box = torch.tensor([[0.5, 0.5, 0.2, 0.2]], requires_grad=True)
    loss = siou_loss(box, box.detach()).mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(box.grad).all()


def test_metrics_perfect_detection():
    detections = [torch.tensor([[0, 0, 10, 10, 0.9, 0]]).numpy()]
    truths = [torch.tensor([[0, 0, 10, 10, 0]]).numpy()]
    result = calculate_map(detections, truths)
    assert result["ap_class_0"] == 1.0
    assert result["map"] == 0.5


def test_dataset_split_exists():
    split = Path("datasets/splits/train.txt")
    root = Path("dataset")
    if split.exists() and root.exists():
        dataset = SHWDDataset(root, split, limit=1)
        image, targets, _ = dataset[0]
        assert image.shape == (3, 608, 608)
        assert targets.shape[1] == 5
