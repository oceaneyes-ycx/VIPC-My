"""Single-GPU training and sample-weighted, complete-set evaluation."""
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from datasets.splits import CATEGORIES
from datasets.vipc import ViPCDataset
from models.model_utils import fps_subsample
from models.vgtnet import VGTNet
from utils.checkpoint import load_checkpoint
from utils.metrics import completion_metrics


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8")


def make_loader(args, split, training, generator=None):
    dataset = ViPCDataset(args.data_root, split, category=args.category, training=training,
                          view_align=args.view_align, seed=args.seed)
    kwargs = {"prefetch_factor": 2} if args.workers else {}
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=training,
                        drop_last=training, num_workers=args.workers, pin_memory=True,
                        worker_init_fn=seed_worker, generator=generator, **kwargs)
    if not len(loader):
        raise ValueError("Empty loader: reduce --batch-size or check the selected category/split")
    return loader


def prepare_batch(batch):
    partial = fps_subsample(batch["partial"].cuda(non_blocking=True).contiguous(), 2048)
    target = fps_subsample(batch["gt"].cuda(non_blocking=True).contiguous(), 2048)
    return partial, batch["image"].cuda(non_blocking=True), target


@torch.no_grad()
def evaluate(model, loader, threshold):
    model.eval()
    totals = {}
    for batch in tqdm(loader, desc="Evaluate"):
        partial, image, target = prepare_batch(batch)
        _, prediction, _, _ = model(partial, image)
        metrics = completion_metrics(prediction, target, threshold).cpu().double().numpy()
        for category, values in zip(batch["category"], metrics):
            if category not in totals:
                totals[category] = [0, np.zeros(3, dtype=np.float64)]
            totals[category][0] += 1
            totals[category][1] += values
    names = {value: key for key, value in CATEGORIES.items()}

    def record(count, values):
        return {"samples": count, "cd_l1_x1000": float(values[0] * 1000),
                "cd_l2_x1000": float(values[1] * 1000), "fscore": float(values[2])}

    per_category = {names[key]: record(count, values / count) for key, (count, values) in sorted(totals.items())}
    count = sum(item[0] for item in totals.values())
    mean = sum(item[1] for item in totals.values()) / count
    macro = np.mean([values / n for n, values in totals.values()], axis=0)
    return {"sample_mean": record(count, mean), "category_mean": record(count, macro),
            "per_category": per_category, "fscore_squared_distance_threshold": threshold}


def save_checkpoint(path, model, optimizer, scheduler, generator, epoch, best, args):
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "epoch": epoch, "best_cd_l2": best,
                "args": vars(args), "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
                    "loader": generator.get_state()}}, path)


def train(args, model):
    output = args.output or Path("outputs") / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    if (output / "config.json").exists() and not args.resume:
        raise FileExistsError(f"Experiment already exists: {output}; use a new --output or --resume")
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = make_loader(args, args.train_split, True, generator)
    val_loader = make_loader(args, args.val_split, False, torch.Generator().manual_seed(args.seed))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    start, best = 0, float("inf")
    if args.init_checkpoint:
        load_checkpoint(model, args.init_checkpoint)
    if args.resume:
        checkpoint = load_checkpoint(model, args.resume)
        required = {"optimizer", "scheduler", "epoch", "best_cd_l2", "rng", "args"}
        if not required <= checkpoint.keys():
            raise ValueError("Resume requires a release checkpoint; use --init-checkpoint for historical weights")
        # Prevent a resumed run from silently changing its data, loss or optimizer settings.
        mutable = {"command", "output", "resume", "init_checkpoint", "epochs", "save_every", "workers", "device"}
        for key, value in checkpoint["args"].items():
            if key not in mutable and str(getattr(args, key)) != str(value):
                raise ValueError(f"Resume setting differs: {key}; previous={value}, requested={getattr(args, key)}")
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start, best = checkpoint["epoch"], checkpoint["best_cd_l2"]
        rng = checkpoint["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        torch.cuda.set_rng_state_all(rng["cuda"])
        generator.set_state(rng["loader"])
    if start >= args.epochs:
        raise ValueError("--epochs must exceed the checkpoint's completed epoch count")
    write_json(output / ("resume-config.json" if args.resume else "config.json"), vars(args))
    for epoch in range(start + 1, args.epochs + 1):
        model.train()
        loss_sum, samples = 0.0, 0
        for batch in tqdm(train_loader, desc=f"Train {epoch}/{args.epochs}"):
            partial, image, target = prepare_batch(batch)
            stages, _, style_loss, _ = model(partial, image)
            loss = model.get_loss(stages, target) + args.style_weight * style_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(partial)
            samples += len(partial)
        learning_rate = optimizer.param_groups[0]["lr"]
        scheduler.step()
        result = evaluate(model, val_loader, args.fscore_threshold)
        metric = result["category_mean"]["cd_l2_x1000"]
        improved = metric < best
        best = min(best, metric)
        report = {"epoch": epoch, "train_loss": loss_sum / samples, "lr": learning_rate, "evaluation": result}
        write_json(output / f"metrics-{epoch:03d}.json", report)
        print(json.dumps({"epoch": epoch, "loss": report["train_loss"], "category_mean": result["category_mean"]}))
        save_checkpoint(output / "last.pth", model, optimizer, scheduler, generator, epoch, best, args)
        if improved:
            save_checkpoint(output / "best.pth", model, optimizer, scheduler, generator, epoch, best, args)
        if epoch % args.save_every == 0:
            save_checkpoint(output / f"epoch-{epoch:03d}.pth", model, optimizer, scheduler, generator, epoch, best, args)


@torch.no_grad()
def infer(args, model):
    if args.output.suffix != ".npy":
        raise ValueError("--output must end in .npy")
    points = np.load(args.partial, allow_pickle=False).astype(np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2048 or not np.isfinite(points).all():
        raise ValueError("--partial must be a finite N x 3 array with N >= 2048")
    with Image.open(args.image) as image:
        view = transforms.ToTensor()(image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR))
    partial = fps_subsample(torch.from_numpy(points).unsqueeze(0).cuda().contiguous(), 2048)
    model.eval()
    _, prediction, _, _ = model(partial, view.unsqueeze(0).cuda())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, prediction[0].cpu().numpy())
    print(f"Saved 2048 completed points to {args.output}")


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("VGTNet requires a CUDA GPU and compiled extensions; see README.md")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = VGTNet().cuda()
    print(f"VGTNet parameters: {sum(p.numel() for p in model.parameters()):,}")
    if args.command == "train":
        train(args, model)
    else:
        load_checkpoint(model, args.checkpoint)
        if args.command == "infer":
            infer(args, model)
        else:
            loader = make_loader(args, args.split, False, torch.Generator().manual_seed(args.seed))
            result = evaluate(model, loader, args.fscore_threshold)
            result["arguments"] = vars(args)
            write_json(args.output, result)
            print(json.dumps(result, indent=2, default=str))
