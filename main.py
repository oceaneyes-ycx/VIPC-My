"""Command line interface for the VGTNet research release."""
import argparse
import math
import os
from pathlib import Path

from datasets.splits import CATEGORIES, SEEN, SPLIT_DIR


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="VGTNet: view-guided point cloud completion")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "test", "infer"):
        command = commands.add_parser(name)
        command.add_argument("--device", default="0", help="one visible CUDA GPU index")
        command.add_argument("--seed", type=int, default=1)
        if name == "infer":
            command.add_argument("--checkpoint", type=Path, required=True)
            command.add_argument("--partial", type=Path, required=True, help="prealigned, normalized N x 3 .npy")
            command.add_argument("--image", type=Path, required=True)
            command.add_argument("--output", type=Path, default=Path("outputs/completion.npy"))
            continue
        command.add_argument("--data-root", type=Path, required=True)
        command.add_argument("--category", choices=("all", *SEEN) if name == "train" else ("all", "unseen", *CATEGORIES), default="all")
        command.add_argument("--workers", type=int, default=8)
        command.add_argument("--batch-size", type=positive_int, default=64 if name == "train" else 8)
        command.add_argument("--view-align", action="store_true", help="use the partial scan's camera view for the image")
        command.add_argument("--fscore-threshold", type=float, default=0.001, help="threshold on squared Euclidean distance")
        if name == "test":
            command.add_argument("--checkpoint", type=Path, required=True)
            command.add_argument("--split", type=Path, default=SPLIT_DIR / "test_list.txt.gz")
            command.add_argument("--output", type=Path, default=Path("outputs/evaluation.json"))
        else:
            command.add_argument("--train-split", type=Path, default=SPLIT_DIR / "train_list.txt.gz")
            command.add_argument("--val-split", type=Path, default=SPLIT_DIR / "test_list.txt.gz")
            command.add_argument("--epochs", type=positive_int, default=200)
            command.add_argument("--lr", type=float, default=0.0005)
            command.add_argument("--style-weight", type=float, default=0.03)
            command.add_argument("--gamma", type=float, default=0.7)
            command.add_argument("--milestones", type=positive_int, nargs="+", default=list(range(20, 201, 20)))
            command.add_argument("--save-every", type=positive_int, default=40)
            command.add_argument("--output", type=Path, help="new experiment directory (default: timestamped under outputs)")
            recovery = command.add_mutually_exclusive_group()
            recovery.add_argument("--resume", type=Path, help="resume a release training checkpoint including optimizer/RNG")
            recovery.add_argument("--init-checkpoint", type=Path, help="initialize model weights, including historical checkpoints")
    args = parser.parse_args(argv)
    if not args.device.isdigit():
        parser.error("--device must identify one CUDA GPU, for example 0")
    if not 0 <= args.seed < 2 ** 32:
        parser.error("--seed must be in [0, 2**32)")
    if args.command != "infer":
        if args.workers < 0:
            parser.error("--workers must be >= 0")
        if not math.isfinite(args.fscore_threshold) or args.fscore_threshold <= 0:
            parser.error("--fscore-threshold must be positive")
    if args.command == "train":
        if not all(math.isfinite(value) for value in (args.lr, args.style_weight, args.gamma)):
            parser.error("learning rate, style weight and gamma must be finite")
        if args.lr <= 0 or args.style_weight < 0 or not 0 < args.gamma <= 1:
            parser.error("require lr > 0, style-weight >= 0 and 0 < gamma <= 1")
    return args


def main(argv=None):
    args = parse_args(argv)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    # Delay ML imports so that --help also works without PyTorch/CUDA installed.
    from run.engine import run
    run(args)


if __name__ == "__main__":
    main()
