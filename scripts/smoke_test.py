"""CUDA forward/backward, metric and checkpoint checks with synthetic inputs."""
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import torch
    from models.vgtnet import VGTNet
    from utils.checkpoint import load_checkpoint
    from utils.metrics import completion_metrics
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required for this smoke test")
    torch.manual_seed(1)
    model = VGTNet().cuda().train()
    points = torch.randn(1, 2048, 3, device="cuda") * 0.2
    image = torch.rand(1, 3, 224, 224, device="cuda")
    stages, prediction, style, _ = model(points, image)
    assert [tuple(item.shape) for item in stages] == [(1, 1024, 3), (1, 4096, 3), (1, 2048, 3)]
    loss = model.get_loss(stages, points) + 0.03 * style
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    identical = completion_metrics(points, points)
    assert torch.allclose(identical, torch.tensor([[0., 0., 1.]], device="cuda"), atol=1e-6)
    # Known translation: squared distance is 0.0004 in each direction.
    a = torch.zeros(1, 1, 3, device="cuda")
    b = torch.tensor([[[0.02, 0., 0.]]], device="cuda")
    assert torch.allclose(completion_metrics(a, b), torch.tensor([[0.02, 0.0008, 1.]], device="cuda"), atol=1e-6)
    assert completion_metrics(a, b, threshold=0.0001)[0, 2] == 0
    with tempfile.TemporaryDirectory() as folder:
        checkpoint = Path(folder) / "checkpoint.pth"
        original = model.lstnet.mlp[0].weight.detach().clone()
        torch.save({"model": {"module." + key: value for key, value in model.state_dict().items()}}, checkpoint)
        with torch.no_grad():
            model.lstnet.mlp[0].weight.zero_()
        load_checkpoint(model, checkpoint)
        assert torch.equal(model.lstnet.mlp[0].weight, original)
    print(f"PASS: stages, finite gradients, metric conventions, strict checkpoint loading; {prediction.shape=}")


if __name__ == "__main__":
    main()
