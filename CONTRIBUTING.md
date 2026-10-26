# Contributing

Keep changes scoped to VGTNet and document any effect on the paper's data or metric protocol. Put unrelated model experiments on separate branches. Do not commit datasets, weights, generated logs, compiled extensions or personal paths.

Before a pull request, run:

```bash
python -m unittest discover -s tests -v
python -m compileall -q main.py datasets models run utils extensions scripts tests
```

For model or CUDA changes, also build the extensions and run `python scripts/smoke_test.py`. Report Python, PyTorch, CUDA, GPU and compiler versions together with the exact command. Distinguish synthetic checks from full dataset evaluation; include split, category, seed and view-alignment settings for benchmark reports.

Maintain existing checkpoint parameter names or document migration explicitly. Preserve third-party notices. Contributions are submitted under the project's MIT license for original contributions, with applicable upstream terms retained.
