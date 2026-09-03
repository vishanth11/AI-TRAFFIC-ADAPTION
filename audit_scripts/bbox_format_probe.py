"""Phase 1 bbox format probe — is objects.bbox [xmin,ymin,xmax,ymax] or [x,y,w,h]?

Reads only the first few row groups per shard; small sample is enough.
"""

from pathlib import Path

import pyarrow.parquet as pq

DATASET_DIR = Path("dataset")
SAMPLE_SHARDS = [
    "train-00000-of-00002.parquet",
    "validation-00000-of-00001.parquet",
]

IMG = 640
MAX_ROW_GROUPS = 3


def main() -> None:
    xyxy_valid = 0          # x2 > x1 and y2 > y1
    xywh_valid = 0          # x + w <= IMG and y + h <= IMG
    unordered = 0
    total = 0
    w_gt_320 = 0            # implausibly wide if format were width

    for fname in SAMPLE_SHARDS:
        pf = pq.ParquetFile(DATASET_DIR / fname)
        for rg in range(min(MAX_ROW_GROUPS, pf.metadata.num_row_groups)):
            tbl = pf.read_row_group(rg, columns=["objects"])
            for obj in tbl.column("objects").to_pylist():
                for b in (obj or {}).get("bbox") or []:
                    x1, y1, b2, b3 = b
                    total += 1
                    if b2 > x1 and b3 > y1:
                        xyxy_valid += 1
                    if x1 + b2 <= IMG and y1 + b3 <= IMG:
                        xywh_valid += 1
                    if b2 <= x1 or b3 <= y1:
                        unordered += 1
                    if b2 > 320:
                        w_gt_320 += 1

    print(f"boxes sampled        : {total}")
    print(f"corner-valid (x2>x1,y2>y1)      : {xyxy_valid} "
          f"({xyxy_valid / total * 100:.1f}%)")
    print(f"xywh-valid (x+w<=640, y+h<=640) : {xywh_valid} "
          f"({xywh_valid / total * 100:.1f}%)")
    print(f"boxes with b2<=x1 or b3<=y1     : {unordered} "
          f"({unordered / total * 100:.1f}%)")
    print(f"boxes with b2>320 (wide 'w'?)   : {w_gt_320}")


if __name__ == "__main__":
    main()