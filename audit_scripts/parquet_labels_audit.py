"""Phase 1 Parquet deep audit — label distribution, bbox format, contamination.

Streams row groups one at a time (max ~100 rows in memory).
Hashes embedded PNG bytes across splits to detect duplicate images.
"""

import hashlib
import struct
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

DATASET_DIR = Path("dataset")
SHARDS = {
    "train": ["train-00000-of-00002.parquet", "train-00001-of-00002.parquet"],
    "validation": ["validation-00000-of-00001.parquet"],
    "test": ["test-00000-of-00001.parquet"],
}


def png_dimensions(data: bytes) -> tuple[int, int] | None:
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    w, h = struct.unpack(">II", data[16:24])
    return w, h


def stream_categories(path: Path):
    """Yield (row_idx, categories, bboxes) per record, one row group at a time."""
    pf = pq.ParquetFile(path)
    idx = 0
    for rg in range(pf.metadata.num_row_groups):
        tbl = pf.read_row_group(rg, columns=["objects"])
        objs = tbl.column("objects").to_pylist()
        for o in objs:
            yield idx, (o or {}).get("category") or [], (o or {}).get("bbox") or []
            idx += 1


def main() -> None:
    hashes: dict[str, list[str]] = {}
    dims_seen: Counter = Counter()

    for split, files in SHARDS.items():
        cat_counter: Counter = Counter()
        objects_per_image: Counter = Counter()
        n_images = 0
        n_objects = 0
        empty_images = 0
        bbox_stats = {"xmin": [], "ymin": [], "xmax": [], "ymax": []}
        img_hashes: list[str] = []

        for fname in files:
            path = DATASET_DIR / fname
            pf = pq.ParquetFile(path)
            for rg in range(pf.metadata.num_row_groups):
                tbl = pf.read_row_group(rg, columns=["image", "objects"])
                images = tbl.column("image").to_pylist()
                objects = tbl.column("objects").to_pylist()
                for img, obj in zip(images, objects):
                    n_images += 1
                    data = (img or {}).get("bytes") or b""
                    if not data:
                        empty_images += 1
                    else:
                        h = hashlib.sha256(data).hexdigest()
                        img_hashes.append(h)
                        if len(dims_seen) < 50:
                            d = png_dimensions(data)
                            if d:
                                dims_seen[d] += 1

                    cats = (obj or {}).get("category") or []
                    boxes = (obj or {}).get("bbox") or []
                    cat_counter.update(cats)
                    objects_per_image[len(cats)] += 1
                    n_objects += len(cats)
                    for b in boxes:
                        bbox_stats["xmin"].append(b[0])
                        bbox_stats["ymin"].append(b[1])
                        bbox_stats["xmax"].append(b[2])
                        bbox_stats["ymax"].append(b[3])

        hashes[split] = img_hashes
        total_imgs_with_cat = sum(objects_per_image.values())
        print(f"\n{'=' * 60}\nSPLIT: {split}\n{'=' * 60}")
        print(f"images           : {n_images}")
        print(f"images missing bytes: {empty_images}")
        print(f"total objects    : {n_objects}")
        print(f"objects/image    : {n_objects / max(n_images, 1):.2f} avg")
        print(f"objects/image dist: {dict(sorted(objects_per_image.items()))}")
        print(f"category counts  : {dict(sorted(cat_counter.items()))}")
        for c, n in sorted(cat_counter.items()):
            print(f"  class {c}: {n} ({n / max(n_objects, 1) * 100:.2f}%)")
        print(f"images with zero objects: {objects_per_image.get(0, 0)}")
        for k, vals in bbox_stats.items():
            if vals:
                print(f"bbox {k}: min={min(vals):.1f} max={max(vals):.1f} "
                      f"mean={sum(vals) / len(vals):.1f}")

    print(f"\n{'=' * 60}\nIMAGE DIMENSIONS (sampled)\n{'=' * 60}")
    print(dict(dims_seen))

    print(f"\n{'=' * 60}\nSPLIT CONTAMINATION (SHA-256 of embedded PNG bytes)\n{'=' * 60}")
    train_set = set(hashes["train"])
    val_set = set(hashes["validation"])
    test_set = set(hashes["test"])
    tv = train_set & val_set
    tt = train_set & test_set
    vt = val_set & test_set
    allh = train_set | val_set | test_set
    print(f"unique images overall     : {len(allh)}")
    print(f"train & validation dupes  : {len(tv)}")
    print(f"train & test dupes        : {len(tt)}")
    print(f"validation & test dupes   : {len(vt)}")
    print(f"train internal dupes      : {len(hashes['train']) - len(train_set)}")
    print(f"validation internal dupes : {len(hashes['validation']) - len(val_set)}")
    print(f"test internal dupes       : {len(hashes['test']) - len(test_set)}")


if __name__ == "__main__":
    main()