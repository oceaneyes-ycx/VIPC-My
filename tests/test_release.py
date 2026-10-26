"""CPU checks for release metadata, CLI contracts and preserved data splits."""
import contextlib
import gzip
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from datasets.splits import CATEGORIES, SEEN, UNSEEN, SPLIT_DIR, category_ids, read_split
from main import parse_args


class SplitTests(unittest.TestCase):
    def test_original_split_content_and_categories(self):
        manifest = json.loads((SPLIT_DIR / "manifest.json").read_text())
        for name, info in manifest.items():
            with self.subTest(name=name):
                path = SPLIT_DIR / (name + ".gz")
                raw = gzip.decompress(path.read_bytes())
                self.assertEqual(hashlib.sha256(raw).hexdigest(), info["sha256"])
                self.assertEqual(len(raw.splitlines()), info["rows"])
                for category in ("all", "unseen"):
                    if "onlyunseen" in name and category == "all":
                        continue
                    rows = read_split(path, category)
                    self.assertTrue(all(row[0] in category_ids(category) for row in rows))

    def test_training_and_unseen_categories_are_disjoint(self):
        self.assertEqual(len(category_ids("all")), 8)
        self.assertEqual(len(category_ids("unseen")), 4)
        self.assertFalse(category_ids("all") & category_ids("unseen"))
        self.assertEqual(category_ids("all"), {CATEGORIES[key] for key in SEEN})
        self.assertEqual(category_ids("unseen"), {CATEGORIES[key] for key in UNSEEN})

    def test_safe_paths_and_camera_range(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "split.txt"
            for row in ("02691156/../00", "02691156/model/24", "99999999/model/00"):
                path.write_text(row)
                with self.assertRaises(ValueError):
                    read_split(path)
            # ShapeNet object IDs include short hashes and UUID-style names.
            path.write_text("02691156/u-sample_1/00\n02691156/abc/23\n")
            self.assertEqual(len(read_split(path)), 2)


class CliTests(unittest.TestCase):
    def test_paper_defaults(self):
        args = parse_args(["train", "--data-root", "data", "--workers", "0"])
        self.assertEqual((args.batch_size, args.epochs, args.lr, args.style_weight), (64, 200, 0.0005, 0.03))
        self.assertEqual(args.category, "all")

    def test_required_checkpoint_and_rejected_unseen_training(self):
        for arguments in (["test", "--data-root", "data"],
                          ["train", "--data-root", "data", "--category", "unseen"],
                          ["train", "--data-root", "data", "--workers", "-1"],
                          ["train", "--data-root", "data", "--batch-size", "0"],
                          ["train", "--data-root", "data", "--style-weight", "nan"],
                          ["train", "--data-root", "data", "--seed", "-1"],
                          ["train", "--data-root", "data", "--device", "0,1"]):
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    parse_args(arguments)
                self.assertEqual(error.exception.code, 2)

    def test_evaluation_allows_unseen(self):
        args = parse_args(["test", "--data-root", "data", "--checkpoint", "weights.pth", "--category", "unseen"])
        self.assertEqual(args.category, "unseen")


if __name__ == "__main__":
    unittest.main()
