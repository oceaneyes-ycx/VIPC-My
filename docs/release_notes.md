# Public release cleanup

Prepared from research commit `efe6f12b43536faca28db3b31339b99b53d6b7b4` on the original `master` branch. The original branch remains the recovery point for all excluded experiments.

## Included

- The VGTNet main architecture from `SymmCompletion3`, keeping checkpoint parameter names.
- Only model-reachable point/attention helpers and two required CUDA extensions.
- A single `main.py train/test/infer` interface with explicit dataset/checkpoint/output paths.
- Original full, reduced and unseen split definitions, compressed without changing their bytes.
- MIT license, third-party acknowledgements and license texts, paper citation metadata, English/Chinese documentation and the paper's pipeline figure.
- CPU checks, CUDA smoke test, data-path preflight and a CPU GitHub Actions workflow.

## Excluded

656 tracked research/artifact files were removed from the release working tree before replacements were added. This includes other model families, duplicated extensions, unused ablations, pretraining/P2C experiments, `old code/`, hard-coded visualization/debug scripts, editor settings, machine-specific caches and binaries, generated point clouds, event logs, ad hoc split variants and a bundled Python source tarball. Files needed by the released model were retained or extracted first. Original experiments remain on `master` rather than being packaged as part of the public code.

The original root split files are now gzip archives under `data/splits/`. Native binaries must be rebuilt for each environment. `.gitignore` excludes environments, local data, weights, logs, build outputs and cleanup scratch files.

## Behavior fixes

Standalone evaluation now creates the correct model and loads the requested checkpoint, includes the last batch and weights averages by sample count. Reports expose per-category and macro averages. No wandb account, remote logging or automatic export of evaluation samples is required.

Resume is explicit and stores optimizer/scheduler/epoch/best metric/RNG state; old weight files can initialize the model with `--init-checkpoint`. Model selection uses category-averaged CD-L2. Dataset and output paths are CLI arguments. Dataloader `--workers 0` no longer passes an invalid prefetch configuration.

All changes to defaults and evaluation behavior are documented in [reproduction.md](reproduction.md). GPU verification remains outstanding and is not represented by the CPU workflow's result.

## GitHub publication

The public branch should be pushed to a **new public repository**, rather than making the research repository public with its historical branches. An isolated initial commit keeps excluded research snapshots out of the new repository's history; do not push `master`, all branches or all tags. The cleanup does not change the existing `origin` URL or push any code automatically.

After checking the GPU smoke test and selecting the destination repository, push the prepared release branch only:

```bash
git push https://github.com/YOUR_ACCOUNT/YOUR_NEW_REPOSITORY.git release/vgtnet-open-source:main
```

Replace the destination with the repository you intend to publish. This example does not publish the existing private research history or change local remote configuration.
