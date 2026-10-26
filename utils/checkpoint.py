"""Load both release and historical SymmCompletion3 checkpoints strictly."""
import torch


def load_checkpoint(model, path):
    # Training checkpoints contain optimizer state: only load files you trust.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    return checkpoint
