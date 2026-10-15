"""RailSem19 loader — primary visual golden dataset for rail-scene segmentation.

Research-only license. Download URL and directory layout captured in
memory-bank/research/03-aws-best-practices-and-golden-datasets.md §2.

This loader expects the dataset unpacked under a local cache dir. v1 loads
a 100-image subset for fast E2E iteration; full dataset is 8,500 images / ~1.7 GB.

Required layout:
    <cache>/rs19_val/
        jpgs/rs19_val/<id>.jpg
        uint8/rs19_val/<id>.png   # class-index masks
        rs19-config.json          # class → id mapping
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

RAILSEM19_BASE_URL = "https://wilddash.cc/railsem19"  # research terms-of-use gated
RAILSEM19_NUM_CLASSES = 19  # per rs19-config.json


@dataclass
class RailSem19Sample:
    image_path: str
    mask_path: str


@dataclass
class RailSem19Dataset:
    root: Path
    samples: list[RailSem19Sample] = field(default_factory=list)
    class_map: dict[int, str] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.samples)


def load_railsem19(root: str | Path, *, subset_size: int | None = 100) -> RailSem19Dataset:
    """Index a local RailSem19 unpacked directory.

    Args:
        root: path to the unpacked `rs19_val` folder.
        subset_size: optional cap (None = all). v1 default is 100 for fast iteration.
    """
    root_p = Path(root)
    if not root_p.exists():
        raise FileNotFoundError(
            f"RailSem19 not found at {root_p}. See memory-bank/research/03-*.md for download."
        )
    img_dir = root_p / "jpgs" / "rs19_val"
    mask_dir = root_p / "uint8" / "rs19_val"
    if not img_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(
            f"RailSem19 subdirectories missing. Expected {img_dir} and {mask_dir}."
        )

    image_paths = sorted(img_dir.glob("*.jpg"))
    samples: list[RailSem19Sample] = []
    for ip in image_paths:
        mp = mask_dir / f"{ip.stem}.png"
        if mp.exists():
            samples.append(RailSem19Sample(image_path=str(ip), mask_path=str(mp)))
        if subset_size is not None and len(samples) >= subset_size:
            break

    class_map: dict[int, str] = {}
    cfg = root_p / "rs19-config.json"
    if cfg.exists():
        try:
            raw: Any = json.loads(cfg.read_text())
            for entry in raw.get("labels", []):
                cid = entry.get("id")
                name = entry.get("name")
                if isinstance(cid, int) and isinstance(name, str):
                    class_map[cid] = name
        except Exception:  # nosec B110
            pass

    return RailSem19Dataset(root=root_p, samples=samples, class_map=class_map)
