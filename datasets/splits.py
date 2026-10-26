"""Split parsing and category selection; this module has no ML dependencies."""
import gzip
import re
from pathlib import Path

CATEGORIES = {
    "plane": "02691156", "bench": "02828884", "cabinet": "02933112",
    "car": "02958343", "chair": "03001627", "monitor": "03211117",
    "lamp": "03636649", "speaker": "03691459", "firearm": "04090263",
    "couch": "04256520", "table": "04379243", "cellphone": "04401088",
    "watercraft": "04530566",
}
SEEN = ("plane", "cabinet", "car", "chair", "lamp", "couch", "table", "watercraft")
UNSEEN = ("bench", "monitor", "speaker", "cellphone")
SPLIT_DIR = Path(__file__).resolve().parents[1] / "data" / "splits"
ROW = re.compile(r"([0-9]{8})/([A-Za-z0-9_-]+)/([0-9]{2})")


def category_ids(category):
    """`all` selects the eight benchmark training categories, excluding unseen ones."""
    names = SEEN if category == "all" else UNSEEN if category == "unseen" else (category,)
    try:
        return {CATEGORIES[name] for name in names}
    except KeyError as exc:
        raise ValueError(f"Unknown category: {category}") from exc


def read_split(path, category="all"):
    """Read category/object/view rows from plain text or a lossless gzip archive."""
    path = Path(path)
    allowed = category_ids(category)
    opener = gzip.open if path.suffix == ".gz" else open
    records = []
    with opener(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            key = line.strip()
            if not key:
                continue
            match = ROW.fullmatch(key)
            if not match or int(match[3]) >= 24 or match[1] not in CATEGORIES.values():
                raise ValueError(f"{path}:{line_number}: invalid split row {key!r}")
            if match[1] in allowed:
                records.append((match[1], match[2], match[3]))
    if not records:
        raise ValueError(f"No samples for category {category!r} in {path}")
    return records
