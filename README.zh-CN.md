# VGTNet：基于几何变换的跨模态点云补全

Chuxiu Yu、Takeshi Oishi，东京大学。论文：[Cross Modal Point Cloud Completion with Geometric Transformation](CGVI2025_Yu.pdf)。

VGTNet 以部分点云和单张 RGB 图像为输入。CGTNet 对 512 个关键点预测几何变换，构造 1024 点粗点云；图像引导的细化模块将其扩展为 4096 点，最终通过 FPS 输出 2048 点。

![论文中的方法流程](assets/pipeline.png)

本分支仅保留论文主模型及必要依赖。主模型在 [models/vgtnet.py](models/vgtnet.py)，来自原 `SymmCompletion3`，保留了参数名以支持旧权重。原 README 中 EGIInet 的介绍、权重链接和默认入口已替换。

## 安装与使用

完整安装命令和目录说明见 [英文 README](README.md)。目标环境是 Linux、单张 NVIDIA GPU、CUDA toolkit 与 C++ 编译器。原仓库给出的 Python 3.9 / PyTorch 2.1.2 / CUDA 11.8 组合仅作为安装起点，本次没有完成该环境的 GPU 复验。

安装依赖后执行：

```bash
python scripts/install_extensions.py
python scripts/smoke_test.py
python scripts/check_data.py --data-root /path/to/ShapeNetViPC-Dataset --category plane
python main.py train --data-root /path/to/ShapeNetViPC-Dataset --category plane --output outputs/plane
python main.py test --data-root /path/to/ShapeNetViPC-Dataset --category plane --checkpoint outputs/plane/best.pth --output outputs/plane-evaluation.json
```

`--category all` 仅包含论文中的 8 个已见类别；`--category unseen` 用于测试 4 个未见类别，训练入口不允许选择它们。`couch` 对应论文的 sofa。`--workers 0` 方便排查数据加载问题，`--device 0` 选择 GPU。

原始 split 已无损压缩保存在 `data/splits/`，附有校验值。原始划分在对象层面存在交集，整理时未擅自替换数据协议。默认图像与点云视角分别选择，测试固定每个样本的随机种子；`--view-align` 改为同视角。

训练默认采用论文中的 batch size 64、200 epochs、学习率 0.0005、风格损失权重 0.03。旧 `trainv8.py` 实际使用的权重是 0.3，且使用了缩小后的 split；完整差异见 [复现说明](docs/reproduction.md)。默认在原测试列表上选择最佳 checkpoint；如需独立验证集，请明确指定 `--val-split`。

本版本没有附带 VGTNet 预训练权重，也没有重新验证论文表格数值。独立测试会严格加载提供的权重，保留最后一个不足 batch 的批次，并输出每类别指标、类别平均值和样本加权平均值。F-score 的 0.001 阈值作用于平方欧氏距离。

## 检查与许可

```bash
python -m unittest discover -s tests -v
python -m compileall -q main.py datasets models run utils extensions scripts tests
```

CPU 检查不需要 PyTorch。GPU smoke test 验证模型各阶段形状、梯度、指标定义与权重读写，需要安装 PyTorch 并编译 CUDA 扩展。

原创代码采用 [MIT](LICENSE)；第三方代码署名与许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。原实验代码可从原 `master` 分支恢复。发布变更见 [整理说明](docs/release_notes.md)。
