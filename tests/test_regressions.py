"""Regression checks for architecture, loss scaling and interrupted training."""

import json
from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

import train
from models.common import ConvBnAct
from models.cspdarknet import CSPDarknet53
from models.yolo import build_model
from models.pp_lcnet import PPLCNet, DepthSepConv, SEModule
from models.pb_module import BiFPN
from datasets.audit import audit_dataset, require_paper_dataset
from utils.colab_sync import load_checkpoint
from utils.loss import ComputeLoss

ANCHORS = [[[12, 16], [19, 36], [40, 28]],
           [[36, 75], [76, 55], [72, 146]],
           [[142, 110], [192, 243], [459, 401]]]


@pytest.mark.parametrize("number", range(1, 7))
def test_all_yaml_models_at_608(number):
    model = build_model(f"configs/model{number}.yaml").eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 608, 608))
    assert [tuple(t.shape) for t in out] == [(1, 21, s, s) for s in (76, 38, 19)]


def test_baseline_backbone_depth_and_activations():
    model = CSPDarknet53()
    assert [len(getattr(model, f"stage{i}").blocks) for i in range(1, 6)] == [1, 2, 8, 8, 4]
    assert model.stem.conv.stride == (1, 1)
    assert all(isinstance(m.act, nn.Mish) for m in model.modules() if isinstance(m, ConvBnAct))


def test_neck_width_configuration_changes_channels():
    model = build_model({"backbone": "pplcnet", "neck": "panet", "dsc": True,
                         "neck_channels": [64, 128, 256]}).eval()
    assert model.neck.out_channels == (32, 64, 128)
    with torch.no_grad():
        assert [t.shape[1] for t in model(torch.zeros(1, 3, 64, 64))] == [21, 21, 21]


def test_paper_backbone_groups_and_neck_widths():
    backbone = PPLCNet(layout="paper", se_blocks=2)
    assert [len(getattr(backbone, f"stage{i}")) for i in (3, 4, 5, 6)] == [2, 2, 2, 7]
    assert sum(isinstance(m, SEModule) for m in backbone.modules()) == 2
    assert all(m.dw.kernel_size == (5, 5) and m.se_after_pw for m in backbone.stage6)
    assert all(m.dw.kernel_size == (3, 3) for m in backbone.stage5)
    for i in (1, 2, 3, 4, 5, 6):
        model = build_model(f"configs/model{i}.yaml")
        assert model.neck.out_channels == (128, 256, 512)
        assert [head[0].pointwise.out_channels if hasattr(head[0], "pointwise")
                else head[0].conv.out_channels for head in model.head.heads] == [256, 512, 1024]


def test_bifpn_matches_figure4_signal_paths():
    # Equal fusion weights and identity transforms make Figure 4 hand-computable.
    fusion = BiFPN((1, 1, 1), dsc=True)
    fusion.down3 = nn.AvgPool2d(2)
    fusion.down4 = nn.AvgPool2d(2)
    for name in ("up5", "up4", "c_mid4", "c_o3", "c_o4", "c_o5"):
        setattr(fusion, name, nn.Identity())
    for name in ("w_mid4", "w_o3", "w_o4", "w_o5"):
        getattr(fusion, name).eps = 0
    out3, out4, out5 = fusion(torch.full((1, 1, 8, 8), 2.),
                             torch.full((1, 1, 4, 4), 6.),
                             torch.full((1, 1, 2, 2), 14.))
    # middle=(2+6)/2=4; coarse=(4+14)/2=9;
    # output middle=(6+4+9)/3; fine=(2+19/3)/2.
    torch.testing.assert_close(out5, torch.full_like(out5, 9.))
    torch.testing.assert_close(out4, torch.full_like(out4, 19/3))
    torch.testing.assert_close(out3, torch.full_like(out3, 25/6))


def test_anchor_matching_searches_all_scales():
    criterion = ComputeLoss(ANCHORS, anchor_iou_threshold=1.0)
    # Size 65 formerly forced scale 1; the closest global anchor is (40,28), scale 0.
    target = torch.tensor([[0, 0, .5, .5, 65/608, 20/608]])
    assignment = criterion.match_anchors(target)
    assert assignment.sum() == 1 and assignment[0, 2]


def test_overlapping_unassigned_box_is_not_a_negative():
    criterion = ComputeLoss(ANCHORS)
    boxes = torch.tensor([[[[[.5, .5, .2, .2], [.9, .9, .1, .1]]]]])
    target = torch.tensor([[0, 0, .5, .5, .2, .2]])
    mask = criterion._objectness_mask(boxes, target)
    assert not mask.flatten()[0] and mask.flatten()[1]


def test_empty_targets_have_finite_loss_and_gradients():
    raw = [torch.zeros(2, 21, s, s, requires_grad=True) for s in (8, 4, 2)]
    loss, parts = ComputeLoss(ANCHORS)(raw, torch.zeros(0, 6))
    assert parts["box"] == 0 and parts["cls"] == 0 and torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(t.grad).all() for t in raw)


def test_dataset_audit_rejects_uncorrected_data(tmp_path):
    root = tmp_path / "dataset"
    (root / "train").mkdir(parents=True)
    (root / "annotations").mkdir()
    (root / "train" / "one.jpg").touch()
    (root / "annotations" / "one.xml").write_text(
        '<annotation><object><name>hat</name></object></annotation>', encoding="utf-8")
    split = tmp_path / "train.txt"
    split.write_text('dataset/train/one.jpg\n', encoding="utf-8")
    report = audit_dataset(root, [split])
    assert report["counts"] == {"images": 1, "hat": 1, "person": 0}
    assert not report["counts_match_paper"]
    with pytest.raises(ValueError, match="cleaned SHWD"):
        require_paper_dataset(root, [split])


@pytest.mark.parametrize("kind", ["ciou", "siou"])
def test_loss_is_invariant_to_repeated_batch(kind):
    raw = [torch.zeros(1, 21, s, s, requires_grad=True) for s in (8, 4, 2)]
    targets = torch.tensor([[0, 0, .4, .4, .04, .06], [0, 1, .6, .6, .2, .3]])
    criterion = ComputeLoss(ANCHORS, loss_type=kind)
    loss, parts = criterion(raw, targets)
    doubled = torch.cat([targets, targets + torch.tensor([1, 0, 0, 0, 0, 0])])
    other, other_parts = criterion([t.repeat(2, 1, 1, 1) for t in raw], doubled)
    assert float(other.detach()) == pytest.approx(float(loss.detach()))
    assert other_parts == pytest.approx(parts)
    loss.backward()
    assert all(torch.isfinite(t.grad).all() for t in raw)


def test_siou_configuration_changes_loss():
    raw = [torch.zeros(1, 21, s, s) for s in (8, 4, 2)]
    targets = torch.tensor([[0, 0, .4, .4, .04, .06]])
    a, _ = ComputeLoss(ANCHORS, loss_type="siou", siou_theta=1.0)(raw, targets)
    b, _ = ComputeLoss(ANCHORS, loss_type="siou", siou_theta=4.0)(raw, targets)
    assert not torch.isclose(a, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_model6_amp_updates_backbone_with_finite_gradients():
    torch.manual_seed(7)
    model = build_model("configs/model6.yaml").cuda().train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", init_scale=1.0,
                                enabled=not torch.cuda.is_bf16_supported())
    before = model.backbone.stem.conv.weight.detach().clone()
    images = torch.randn(2, 3, 608, 608, device="cuda")
    targets = torch.tensor([[0, 0, .4, .4, .04, .06], [1, 1, .6, .6, .2, .3]], device="cuda")
    with train._autocast(torch.device("cuda"), True):
        loss, _ = ComputeLoss(ANCHORS, loss_type="siou")(model(images), targets)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    scaler.step(optimizer)
    scaler.update()
    assert not torch.equal(before, model.backbone.stem.conv.weight)


@pytest.mark.parametrize("metadata", [{"best_map": -3.0}, {"best_val_loss": 3.0}])
def test_checkpoint_best_loss_is_positive(tmp_path, metadata):
    model = nn.Linear(2, 2)
    path = tmp_path / "checkpoint.pth"
    torch.save({"model": model.state_dict(), "epoch": 4, **metadata}, path)
    epoch, best = load_checkpoint(path, model)
    assert epoch == 5 and best == 3.0
    assert 2.0 < best


class TinyBackbone(nn.Sequential):
    def load_pretrained(self, path):
        self.load_state_dict(torch.load(path, map_location="cpu"))


class TinyDetector(nn.Module):
    """Small differentiable detector to exercise the real training/checkpoint loop."""

    def __init__(self):
        super().__init__()
        self.backbone = TinyBackbone(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4))
        self.head = nn.Conv2d(4, 21, 1)

    def forward(self, x):
        x = self.head(self.backbone(x))
        return [nn.functional.adaptive_avg_pool2d(x, (s, s)) for s in (4, 2, 1)]

    def get_model_size_mb(self):
        return sum(p.numel() for p in self.parameters()) * 4 / 1024**2


def training_fixture(tmp_path, monkeypatch):
    config = {"model_id": "tiny", "architecture": {"loss": "siou"}, "training": {
        "protocol": "development", "epochs_freeze": 1, "epochs_unfreeze": 2, "lr_freeze": .001, "lr_unfreeze": .0001,
        "batch_size_freeze": 2, "batch_size_unfreeze": 2, "save_interval_epochs": 1}}
    cfg_path, data_path = tmp_path / "model.yaml", tmp_path / "data.yaml"
    cfg_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    data_path.write_text(yaml.safe_dump({"img_size": [32, 32], "anchors": ANCHORS,
                                       "train_split": "unused", "val_split": "unused"}), encoding="utf-8")
    torch.manual_seed(1)
    backbone_path = tmp_path / "backbone.pth"
    torch.save(TinyDetector().backbone.state_dict(), backbone_path)
    images = torch.linspace(0, 1, 2 * 3 * 32 * 32).reshape(2, 3, 32, 32)
    targets = torch.tensor([[0, 0, .4, .4, .2, .2], [1, 1, .6, .6, .3, .3]])
    monkeypatch.setattr(train, "build_model", lambda *a, **kw: TinyDetector())
    monkeypatch.setattr(train, "build_dataloader", lambda *a, **kw: [(images, targets, ["a", "b"])])
    return cfg_path, data_path, backbone_path


@pytest.mark.parametrize("saved_epoch", [0, 1])
def test_resume_matches_continuous_training(tmp_path, monkeypatch, saved_epoch):
    cfg, data, pretrained = training_fixture(tmp_path, monkeypatch)
    snapshots = {}
    real_save = train.save_checkpoint

    def capture(state, *args, **kwargs):
        path = tmp_path / f"epoch_{state['epoch']}.pth"
        torch.save(state, path)
        snapshots[state["epoch"]] = path
        return real_save(state, *args, **kwargs)

    monkeypatch.setattr(train, "save_checkpoint", capture)
    uninterrupted = train.train_model(cfg, data, "cpu", output=tmp_path / "full", backbone_weights=pretrained)
    expected = torch.load(tmp_path / "full" / "last.pth", map_location="cpu")
    assert expected["phase"] == "unfreeze" and expected["freeze_epochs"] == 1
    # Freeze phase must not update BatchNorm statistics.
    first = torch.load(snapshots[0], map_location="cpu")
    assert first["model"]["backbone.1.num_batches_tracked"].item() == 0
    monkeypatch.setattr(train, "save_checkpoint", real_save)
    resumed = train.train_model(cfg, data, "cpu", output=tmp_path / "resumed", resume=snapshots[saved_epoch])
    actual = torch.load(tmp_path / "resumed" / "last.pth", map_location="cpu")
    for key, value in expected["model"].items():
        torch.testing.assert_close(actual["model"][key], value)
    assert resumed["best_val_loss"] == pytest.approx(uninterrupted["best_val_loss"])
    assert len(json.loads(Path(resumed["history"]).read_text())) == 3
    assert actual["scheduler"] == expected["scheduler"]


def test_scratch_backbone_trains_and_resume_keeps_directory(tmp_path, monkeypatch):
    cfg, data, _ = training_fixture(tmp_path, monkeypatch)
    train.train_model(cfg, data, "cpu", output=tmp_path / "scratch", epochs=1)
    payload = torch.load(tmp_path / "scratch" / "last.pth", map_location="cpu")
    assert payload["freeze_epochs"] == 0 and payload["phase"] == "unfreeze"
    assert payload["model"]["backbone.1.num_batches_tracked"].item() > 0
    resumed = train.train_model(cfg, data, "cpu", resume=tmp_path / "scratch" / "last.pth", epochs=2)
    assert Path(resumed["history"]).parent == tmp_path / "scratch"


def test_paper_training_requires_pretrained(tmp_path, monkeypatch):
    cfg, data, _ = training_fixture(tmp_path, monkeypatch)
    config = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    config["training"]["protocol"] = "paper"
    cfg.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="pretrained backbone"):
        train.train_model(cfg, data, "cpu", output=tmp_path / "paper")
    result = train.train_model(cfg, data, "cpu", output=tmp_path / "development", epochs=1, from_scratch=True)
    assert result["protocol"] == "development" and result["freeze_epochs"] == 0
