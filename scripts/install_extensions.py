"""Build only the two CUDA extensions required by VGTNet."""
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    import torch
    from torch.utils.cpp_extension import CUDA_HOME
    if CUDA_HOME is None or not (Path(CUDA_HOME) / "bin" / "nvcc").exists():
        raise SystemExit("CUDA toolkit/nvcc not found; set CUDA_HOME to your toolkit installation")
    if not torch.cuda.is_available() and not os.environ.get("TORCH_CUDA_ARCH_LIST"):
        raise SystemExit("No CUDA GPU visible; for a build host set TORCH_CUDA_ARCH_LIST explicitly")
    if not shutil.which("c++"):
        raise SystemExit("A C++ compiler is required (Linux: install build-essential)")
    root = Path(__file__).resolve().parents[1]
    for relative in ("extensions/Pointnet2/pointnet2", "extensions/chamfer_dist"):
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-build-isolation", "--no-deps", "."],
                       cwd=root / relative, check=True)
    print("Built pointnet2_cuda and chamfer. Next: python scripts/smoke_test.py")


if __name__ == "__main__":
    main()
