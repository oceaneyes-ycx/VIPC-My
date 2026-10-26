"""Check split syntax, categories and dataset file paths without ML dependencies."""
import argparse
import json
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.splits import CATEGORIES, SPLIT_DIR, read_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=SPLIT_DIR / "test_list.txt.gz")
    parser.add_argument("--category", choices=("all", "unseen", *CATEGORIES), default="all")
    parser.add_argument("--limit", type=int, default=10, help="rows to check; 0 checks all rows")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be >= 0")
    records = read_split(args.split, args.category)
    selected = records[:args.limit] if args.limit else records
    missing = set()
    checked_objects = set()
    for category, object_id, view in selected:
        relative = Path(category) / object_id
        paths = [args.data_root / "ShapeNetViPC-Partial" / relative / f"{view}.dat"]
        if (category, object_id) not in checked_objects:
            checked_objects.add((category, object_id))
            rendering = args.data_root / "ShapeNetViPC-View" / relative / "rendering"
            paths.append(rendering / "rendering_metadata.txt")
            for index in range(24):
                paths.extend([rendering / f"{index:02d}.png",
                              args.data_root / "ShapeNetViPC-GT" / relative / f"{index:02d}.dat"])
        missing.update(str(path) for path in paths if not path.is_file())
    print(json.dumps({"selected_rows": len(records), "checked_rows": len(selected),
                      "category_rows": dict(Counter(row[0] for row in records)),
                      "missing_files": len(missing), "missing_examples": sorted(missing)[:20]}, indent=2))
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
