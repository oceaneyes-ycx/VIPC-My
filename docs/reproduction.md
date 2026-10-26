# Reproduction and implementation notes

## Paper to code mapping

The source of truth for the method is [CGVI2025_Yu.pdf](../CGVI2025_Yu.pdf), particularly Sections 3 and 4.3. This release extracts the matching main model from `Symm/symm.py::SymmCompletion3` in the original research snapshot.

| Paper component | Release implementation | Historical name |
| --- | --- | --- |
| VGTNet | `models/vgtnet.py::VGTNet` | `SymmCompletion3` |
| CGTNet | `CSTNet`, attribute `lstnet` | `CSTNet` |
| RGB encoder | `ImageEncoder2` | same |
| Multi-source refinement | `SGFormer2`, attribute `sgformer_1` | same |
| Missing-point encoder | `local_encoder` | same |
| Cross/self-attention fusion | `Fusion`, `CrossFormer` | same |
| Point hierarchy | `models/model_utils.py` | selected original definitions |

The coarse-to-fine model returns `(stages, complete, style_loss, intermediates)`. `stages` has point counts 1024, 4096 and 2048; `complete` has shape `B × 2048 × 3`. Input is `B × 2048 × 3` plus an RGB image `B × 3 × 224 × 224`.

Submodule names, including the historical `corssformer_*` spelling and unused registered refinement layers, are retained for state-dict compatibility. Checkpoint loading strips a leading `module.` from DataParallel checkpoints and then uses `strict=True`. Formats accepted for model initialization/test are `{'model': state_dict}`, `{'state_dict': state_dict}`, or a raw state dict. Other model families, including EGIInet and older SymmCompletion variants, are not interchangeable.

Point FPS now uses the PointNet++ extension already required by the encoder. The old stand-alone FPS CUDA kernel has the same sampling implementation; only its include/helper definitions differ. This removes duplicate binary builds. It does not constitute a CUDA numerical parity check.

## Defaults and historical differences

| Setting | Release default | Historical research snapshot |
| --- | --- | --- |
| Main model | VGTNet (original SymmCompletion3) | `main.py` ran EGIInet; `trainv8.py` ran SymmCompletion3 |
| Style coefficient | 0.03 (paper, Section 4.3) | `30 * 1e-2` = 0.3 in `trainv8.py` |
| Training split | complete original train list | `train_reduced.txt` in `trainv8.py` |
| Evaluation split | complete original test list | reduced list during `trainv8.py` |
| `all` training categories | eight seen categories | loader included all 13 categories in lists |
| Best-checkpoint criterion | mean category CD-L2 | CD-L1 in `testv1.py` |
| Evaluation iteration | ordered, all samples | shuffled, `drop_last=True` |
| Evaluation views | deterministic independent image view per row/seed | global random image view |
| Checkpoint recovery | explicit, strict, full state for release checkpoints | script-dependent, sometimes automatic and incomplete |
| Logging | local JSON and stdout | wandb/TensorBoard and user-specific paths |

The release retains the original model's Chamfer implementation: each stage's Euclidean Chamfer loss is **half** the two directional means. The paper's Eq. (7) displays their sum. Consequently, the relative numerical scaling between the paper formula and code is not identical; the implementation is preserved and this difference must be considered when interpreting a style coefficient.

The model's original Gram style loss also uses an elementwise mean followed by division by `128 * 512`. It is retained verbatim rather than re-derived from the paper's notation. The coarse affine parameters are produced by one 12-output MLP split into a 3×3 matrix and translation; the paper describes this as two heads. The ResNet18 encoder starts without ImageNet pretrained weights, as in the research code.

For a configuration closer to the old `trainv8.py`, explicitly select the reduced splits and 0.3 coefficient:

```bash
python main.py train --data-root /path/to/ShapeNetViPC-Dataset --category plane --style-weight 0.3 --train-split data/splits/train_reduced.txt.gz --val-split data/splits/test_reduced.txt.gz --output outputs/plane-historical
```

This recreates those settings, not the old evaluator's dropped batches or incomplete checkpoint state. The original one-epoch warmup used multiplier 1 and provided no learning-rate ramp; release scheduling uses MultiStepLR directly. No gradient clipping is applied: the original clipping call happened after optimizer updates, so it did not constrain the updates. CLI choices and saved configurations make these differences explicit.

## Data protocol

The split archives decompress to the exact bytes from original commit `efe6f12b43536faca28db3b31339b99b53d6b7b4`. `data/splits/manifest.json` records uncompressed checksums, sizes and row counts. They are preserved inputs, not newly generated splits.

The original full lists contain 735,858 train rows and 183,966 test rows across 13 categories. They have 38,148 object IDs in common when grouped by `(category, object)`; rows identify views, and this snapshot must not be described as an object-disjoint benchmark. Confirm your comparison method's protocol before publishing new numbers. `test_onlyunseen.txt.gz` contains the four unseen categories from the original test list. Category filters exclude those classes from default training regardless of which list is supplied.

By default, each row's partial scan view is fixed, while the image and its GT use a separately selected camera. Partial points are rotated with the original x/y camera convention into that image frame. GT centering and maximum Euclidean radius normalize both clouds; this GT-dependent benchmark preprocessing is not a deployment-time pose/scale estimator. Images use RGB values in [0, 1] without ImageNet normalization.

Partial clouds shorter than 3500 points are repeated with the original rule before FPS. Overlong partial arrays fail explicitly instead of being silently truncated; adjust/preprocess a nonstandard dataset deliberately. GT arrays need at least 2048 points and compatible lengths for batching. Evaluation fixes image choice by `seed + row index`; retain list order, category selection, seed and `--view-align` when comparing runs.

The default `--val-split` uses the original test list to reflect historical model selection. To make claims based on an independent final test set, select a separate validation split during training and keep the final test set for evaluation. The cleanup does not invent an unreported validation split.

## Metrics

- CD-L1: `(mean(sqrt(d_pred)) + mean(sqrt(d_gt))) / 2`, reported ×1000.
- CD-L2: `mean(d_pred) + mean(d_gt)`, reported ×1000.
- F-score: harmonic mean of fractions below 0.001, where `d_pred` and `d_gt` are **squared Euclidean distances**. A zero precision/recall pair gives zero.

Each metric is calculated per sample. JSON reports per-category means, `sample_mean` weighted by number of samples, and `category_mean` averaging categories equally. The latter is the intended aggregate for comparison with the paper tables. The F-score threshold is adjustable and saved in the result file.

## Validation status

During cleanup, six standard-library CPU tests pass: original split checksums/parsing, safe row syntax, seen/unseen category separation, CLI defaults and validation. Python compilation and all CLI help entry points are also checked. These checks do not require the dataset or PyTorch.

The cleanup environment has Python 3.13 and no PyTorch installed. CUDA compilation, real checkpoint compatibility, forward/backward GPU execution and full benchmark reproduction have **not** been run. The CUDA smoke test is supplied in `scripts/smoke_test.py`; it checks stage shapes, finite gradients, known metric values and strict checkpoint round trips. Run it after compiling both extensions.

No pretrained VGTNet weights, complete per-category training configurations or training datasets were present. The source model mapping is based on the architecture and training script; model parameter naming is preserved, but loading an author's real checkpoint remains a required GPU validation step. Paper numbers are clearly labelled as reported results throughout this release.
