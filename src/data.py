"""
CIFAKE dataset access and integrity verification.

The dataset is 120,000 32x32 RGB images, evenly split between photographs drawn
from CIFAR-10 and synthetic images produced by Stable Diffusion v1.4. It is not
stored in this repository; set CIFAKE_ROOT or pass --root to point at a local
copy.

    python -m src.data --root C:/DL_data/cifake --verify --out results/dataset_report.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

CLASS_DIRS = {"REAL": 0, "FAKE": 1}
SPLITS = ("train", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


#: CIFAR-10 categories. CIFAKE encodes an image's category in its file name:
#: "1234 (k).jpg" for category k = 2..10 and "1234.jpg" for category 1. Both the
#: photographs and the generated images follow this scheme.
CATEGORIES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
CATEGORY_GROUPS = {
    "vehicles": ("airplane", "automobile", "ship", "truck"),
    "animals": ("bird", "cat", "deer", "dog", "frog", "horse"),
}


def category_of(path) -> str:
    import re

    m = re.search(r"\((\d+)\)", Path(path).stem)
    return CATEGORIES[(int(m.group(1)) if m else 1) - 1]


def parse_categories(spec: str | None):
    """Accept a group name ("vehicles") or a comma-separated list of categories."""
    if not spec:
        return None
    if spec in CATEGORY_GROUPS:
        return set(CATEGORY_GROUPS[spec])
    names = {c.strip() for c in spec.split(",") if c.strip()}
    unknown = names - set(CATEGORIES)
    if unknown:
        raise SystemExit(f"unknown categories {sorted(unknown)}; expected {CATEGORIES} or {sorted(CATEGORY_GROUPS)}")
    return names


def resolve_root(root=None) -> Path:
    candidate = root or os.environ.get("CIFAKE_ROOT")
    if not candidate:
        raise SystemExit("No dataset root. Pass --root or set CIFAKE_ROOT.")
    p = Path(candidate).expanduser().resolve()
    if not p.is_dir():
        raise SystemExit(f"Dataset root does not exist: {p}")
    return p


def list_split(root: Path, split: str):
    out = {}
    for cls in CLASS_DIRS:
        d = root / split / cls
        if not d.is_dir():
            raise SystemExit(f"Missing expected directory: {d}")
        out[cls] = sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    return out


def inventory(root: Path) -> dict:
    report = {"root": str(root), "splits": {}}
    for split in SPLITS:
        files = list_split(root, split)
        report["splits"][split] = {cls: len(paths) for cls, paths in files.items()}
        report["splits"][split]["total"] = sum(len(v) for v in files.values())
    return report


def check_overlap(root: Path) -> dict:
    """Content-hash train against test.

    Duplicate images spanning the split would inflate every reported score, so
    this is checked before any model is trained rather than assumed.
    """
    digests = {}
    within = {}
    for split in SPLITS:
        seen = []
        for cls, paths in list_split(root, split).items():
            for p in paths:
                seen.append(hashlib.md5(p.read_bytes()).hexdigest())
        digests[split] = set(seen)
        within[split] = len(seen) - len(set(seen))
    shared = digests["train"] & digests["test"]
    return {
        "unique_train": len(digests["train"]),
        "unique_test": len(digests["test"]),
        "duplicates_within_train": within["train"],
        "duplicates_within_test": within["test"],
        "exact_duplicates_across_split": len(shared),
    }


def image_properties(root: Path, sample: int = 500) -> dict:
    from PIL import Image

    sizes, modes, fmts = Counter(), Counter(), Counter()
    for split in SPLITS:
        for cls, paths in list_split(root, split).items():
            for p in paths[:sample]:
                with Image.open(p) as im:
                    sizes[im.size] += 1
                    modes[im.mode] += 1
                    fmts[im.format] += 1
    return {
        "sizes": {f"{w}x{h}": n for (w, h), n in sizes.most_common()},
        "modes": dict(modes.most_common()),
        "formats": dict(fmts.most_common()),
        "sampled_per_class_per_split": sample,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect and verify the CIFAKE dataset.")
    ap.add_argument("--root", default=None)
    ap.add_argument("--verify", action="store_true", help="run the duplicate-overlap check")
    ap.add_argument("--out", default=None, help="write the report to this JSON file")
    args = ap.parse_args()

    root = resolve_root(args.root)
    report = inventory(root)
    report["image_properties"] = image_properties(root)
    if args.verify:
        report["integrity"] = check_overlap(root)

    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"\nwritten -> {args.out}")


if __name__ == "__main__":
    main()
