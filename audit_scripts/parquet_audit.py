"""Phase 1 Parquet audit — traffic accident CCTV dataset (read-only).

Memory-efficient: metadata/schema first, then limited row-group reads.
Never loads a whole shard into memory.
"""

from pathlib import Path

import pyarrow.parquet as pq

DATASET_DIR = Path("dataset")
SHARDS = [
    "train-00000-of-00002.parquet",
    "train-00001-of-00002.parquet",
    "validation-00000-of-00001.parquet",
    "test-00000-of-00001.parquet",
]
SAMPLE_ROWS = 5


def fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def describe_value(v) -> str:
    if isinstance(v, bytes):
        return f"<bytes, {len(v)} bytes, head={v[:16]!r}>"
    if isinstance(v, str):
        return f"<str, {len(v)} chars, head={v[:60]!r}>"
    if isinstance(v, (list,)):
        inner = describe_value(v[0]) if v else "empty"
        return f"<list len={len(v)} of {inner}>"
    if isinstance(v, dict):
        return "<dict " + ", ".join(f"{k}:{describe_value(x)}" for k, x in v.items()) + ">"
    return repr(v)


def audit_shard(path: Path) -> dict:
    print(f"\n{'=' * 72}\n{path.name}\n{'=' * 72}")
    pf = pq.ParquetFile(path)
    md = pf.metadata
    schema = pf.schema_arrow

    print(f"file size        : {fmt_bytes(path.stat().st_size)}")
    print(f"rows             : {md.num_rows}")
    print(f"row groups       : {md.num_row_groups}")
    print(f"columns          : {md.num_columns}")
    print(f"created by       : {md.created_by}")

    print("\n-- schema --")
    for i, field in enumerate(schema):
        print(f"  [{i}] {field.name}: {field.type}")

    # first rows from the first row group only
    n_first = min(SAMPLE_ROWS, md.num_rows)
    tbl = pf.read_row_group(0, columns=None)
    tbl = tbl.slice(0, n_first)

    print(f"\n-- first {n_first} records (structural preview, media truncated) --")
    for r in range(tbl.num_rows):
        print(f"\nrecord {r}:")
        for col in schema.names:
            v = tbl.column(col)[r].as_py()
            print(f"  {col} = {describe_value(v)}")

    # row-group size stats for memory planning
    rg_rows = [md.row_group(i).num_rows for i in range(md.num_row_groups)]
    print(f"\nrow-group row counts: min={min(rg_rows)}, max={max(rg_rows)}, "
          f"n={len(rg_rows)}")

    return {
        "name": path.name,
        "rows": md.num_rows,
        "row_groups": md.num_row_groups,
        "columns": list(schema.names),
        "schema": str(schema),
        "size": path.stat().st_size,
    }


def main() -> None:
    info = {}
    for name in SHARDS:
        p = DATASET_DIR / name
        if not p.exists():
            print(f"MISSING: {p}")
            continue
        info[name] = audit_shard(p)

    # split schema compatibility
    print(f"\n{'=' * 72}\nSPLIT SCHEMA COMPATIBILITY\n{'=' * 72}")
    schemas = {k: v["schema"] for k, v in info.items()}
    uniq = {}
    for k, s in schemas.items():
        uniq.setdefault(s, []).append(k)
    if len(uniq) == 1:
        print("All 4 shards share an IDENTICAL Arrow schema -> compatible.")
    else:
        print(f"WARNING: {len(uniq)} distinct schemas found:")
        for s, files in uniq.items():
            print(f"\nfiles: {files}\n{s}")


if __name__ == "__main__":
    main()