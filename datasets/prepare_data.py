"""Prepare SHWD archives and deterministic train/val/test split files."""

from __future__ import annotations

import argparse
import random
import zipfile
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def extract_zip(zip_path: str | Path, output_dir: str | Path, overwrite: bool = False) -> int:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        members = archive.infolist()
        for member in members:
            target = (output / member.filename).resolve()
            if output.resolve() not in target.parents and target != output.resolve():
                raise ValueError(f"Unsafe archive member: {member.filename}")
            if overwrite or not target.exists():
                archive.extract(member, output)
    return len(members)


def create_splits(data_root: str | Path, split_dir: str | Path,
                  val_ratio: float = 0.1, test_ratio: float = 0.1,
                  seed: int = 42) -> dict[str, int]:
    data_root = Path(data_root)
    split_dir = Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    existing_dirs = {name: data_root / name for name in ("train", "val", "test")}
    if all(directory.is_dir() for directory in existing_dirs.values()):
        grouped = {
            name: sorted(
                p.relative_to(data_root.parent).as_posix()
                for p in directory.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )
            for name, directory in existing_dirs.items()
        }
        for name, values in grouped.items():
            (split_dir / f"{name}.txt").write_text("\n".join(values) + "\n", encoding="utf-8")
            (split_dir / f"{name}_ids.txt").write_text(
                "\n".join(Path(v).stem for v in values) + "\n", encoding="utf-8"
            )
        return {name: len(values) for name, values in grouped.items()}
    images = sorted(
        p.relative_to(data_root.parent).as_posix()
        for p in data_root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        and "annotations" not in p.parts
    )
    if not images:
        raise RuntimeError(f"No images found under {data_root}")
    rng = random.Random(seed)
    rng.shuffle(images)
    test_count = int(round(len(images) * test_ratio))
    val_count = int(round(len(images) * val_ratio))
    test = images[:test_count]
    val = images[test_count:test_count + val_count]
    train = images[test_count + val_count:]
    for name, values in (("train", train), ("val", val), ("test", test)):
        (split_dir / f"{name}.txt").write_text("\n".join(values) + "\n", encoding="utf-8")
        (split_dir / f"{name}_ids.txt").write_text(
            "\n".join(Path(v).stem for v in values) + "\n", encoding="utf-8"
        )
    return {"train": len(train), "val": len(val), "test": len(test)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SHWD data and deterministic splits")
    parser.add_argument("--zip", dest="zip_path")
    parser.add_argument("--data-root", default="dataset")
    parser.add_argument("--split-dir", default="datasets/splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    args = parser.parse_args()
    if args.zip_path:
        print(f"Extracted members: {extract_zip(args.zip_path, args.data_root)}")
    print(create_splits(args.data_root, args.split_dir, args.val_ratio, args.test_ratio, args.seed))


if __name__ == "__main__":
    main()
