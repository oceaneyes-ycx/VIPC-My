# Third-party acknowledgements and licenses

The root MIT license applies to original VGTNet contributions. Existing copyright notices and licenses of derived code remain applicable. Dataset access, the bundled paper and separately obtained weights have their own terms.

| Source | Retained/derived components | Notice |
| --- | --- | --- |
| [SymmCompletion](https://github.com/HKUST-SAIL/SymmCompletion) | geometric transformation, local encoder, refinement and point utility foundations adapted for image guidance | [MIT, © 2025 HongyuYann](licenses/SymmCompletion-MIT.txt) |
| [SVDFormer / PointSea](https://github.com/czvvd/SVDFormer_PointSea) | point abstraction/Transformer helpers inherited through the research code | [MIT, © 2023 Czvvd](licenses/SVDFormer-MIT.txt) |
| [EGIInet](https://github.com/WHU-USI3DV/EGIInet) | research framework lineage, cross-modal style-loss inspiration and ViPC loading/evaluation conventions | upstream [README declares MIT](https://github.com/WHU-USI3DV/EGIInet#license); authors [Hang Xu, Chen Long, Wenxiao Zhang, Yuan Liu, Zhen Cao, Zhen Dong and Bisheng Yang](https://arxiv.org/abs/2407.02887) |
| [ViPC](https://github.com/Hydrogenion/ViPC) | ShapeNet-ViPC dataset protocol and data-loading conventions | [MIT code notice, © 2021 Hydrogenion](licenses/ViPC-MIT.txt); dataset terms remain separate |
| [Pointnet2.PyTorch](https://github.com/sshaoshuai/Pointnet2.PyTorch) | `extensions/Pointnet2/pointnet2/` CUDA source and autograd wrappers | [MIT, © 2019 Shaoshuai Shi](extensions/Pointnet2/LICENSE) |
| [ChamferDistancePytorch](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch), GRNet-derived wrapper | `extensions/chamfer_dist/` | [MIT, © 2019 ThibaultGROUEIX](extensions/chamfer_dist/LICENSE); original Haozhe Xie/Thibault Groueix source headers retained |

Additional full MIT texts for SymmCompletion, SVDFormer and ViPC were preserved from the corresponding upstream LICENSE files during cleanup. The original local PointNet++ and Chamfer license texts are retained verbatim. EGIInet's README declares MIT; no standalone LICENSE text was available in the upstream path reviewed, so no new copyright attribution is substituted for its authors' own notice.

The research repository previously included experiments based on XMFNet, Meta-Transformer, P2C and other utilities. Their experiment implementations and unused vendored packages are excluded from this public snapshot; acknowledging them does not claim authorship of those projects.

Runtime dependencies such as PyTorch, torchvision, NumPy, Pillow, tqdm and ninja are installed separately and retain their respective licenses.
