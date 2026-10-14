"""Read-only verification of the cleaned SHWD counts reported in Section 4.1."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET

import yaml

from .shwd import SHWDDataset, _read_split

PAPER_COUNTS = {"images": 7004, "hat": 7709, "person": 101174}


def audit_dataset(data_root: str | Path, splits: list[str | Path]) -> dict:
    seen: set[str] = set()
    counts: Counter = Counter()
    split_counts = {}
    missing_images, missing_labels, duplicates, unsupported_labels = [], [], [], []
    for split in splits:
        entries = _read_split(split)
        dataset = SHWDDataset(data_root, split_file=split)
        if len(dataset.items) != len(entries):
            missing_images.append(str(split))
        split_counts[Path(split).stem] = len(dataset.items)
        for image in dataset.items:
            image_id = image.stem
            if image_id in seen:
                duplicates.append(image_id)
                continue
            seen.add(image_id)
            label_path = dataset._annotation_path(image)
            if label_path is None or label_path.suffix.lower() != ".xml":
                missing_labels.append(image_id)
                continue
            for obj in ET.parse(label_path).getroot().findall("object"):
                label = (obj.findtext("name") or "").strip().lower()
                if label in {"hat", "person"}:
                    counts[label] += 1
                else:
                    unsupported_labels.append(label)
    measured = {"images": len(seen), "hat": counts["hat"], "person": counts["person"]}
    return {"counts": measured, "expected": PAPER_COUNTS, "splits": split_counts,
            "counts_match_paper": measured == PAPER_COUNTS,
            "missing_image_splits": missing_images, "missing_labels": missing_labels,
            "duplicate_ids": duplicates, "unsupported_labels": dict(Counter(unsupported_labels)),
            "source_identity_verified": False}


def require_paper_dataset(data_root: str | Path, splits: list[str | Path]) -> dict:
    report = audit_dataset(data_root, splits)
    if not report["counts_match_paper"] or any(report[k] for k in (
        "missing_image_splits", "missing_labels", "duplicate_ids", "unsupported_labels"
    )):
        raise ValueError(
            f"Paper protocol requires cleaned SHWD counts {PAPER_COUNTS}; found {report['counts']}. "
            "Use the authors' corrected data, or --dry-run/--from-scratch for development. "
            "Do not remove arbitrary images or labels merely to match the counts."
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="configs/dataset.yaml")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    cfg_path = Path(args.data)
    cfg = yaml.safe_load((cfg_path if cfg_path.is_absolute() else root / cfg_path).read_text(encoding="utf-8"))
    def resolve(path):
        p = Path(path)
        return p if p.is_absolute() else root / p
    report = audit_dataset(resolve(cfg["data_dir"]), [resolve(cfg[k]) for k in ("train_split", "val_split", "test_split")])
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
