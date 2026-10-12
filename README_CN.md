# ENRC code-faithful 兼容版本

本仓库是 ENRC 作者实现的现代环境兼容版本，不是 ClustPy ENRC。
它保留原始 FastAI-v1 卷积自编码器、NrKMeans 初始化和 ENRC 优化行为，
只修改当前 Python、PyTorch、NumPy 和 scikit-learn 已不再支持的接口。

原始来源：

```text
https://gitlab.cs.univie.ac.at/lukas/enrcpublic
commit 6de4f30d38f82a609e7b58c0b62b822750870111
```

本仓库是独立的兼容与复现项目，不是 ENRC 官方仓库，也不由原作者维护。

## 1. 安装

在仓库根目录运行：

```powershell
py -3.11 -m venv .venv-enrc
.\.venv-enrc\Scripts\python.exe -m pip install --upgrade pip
.\.venv-enrc\Scripts\python.exe -m pip install -r requirements.txt
```

测试过的主要版本为 Python 3.11.9、PyTorch 2.11.0 和 FastAI 1.0.61。
这里使用的是 FastAI v1，不要替换成 FastAI v2。

## 2. 准备数据

原始数据发布于：

```text
https://figshare.com/articles/dataset/ENRC/12272921
```

解压后目录应为：

```text
data/cmnist/
data/nr_objects/
data/stickfigures/
```

数据、模型和训练结果不会上传到 GitHub。

## 3. C-MNIST

预训练作者的卷积残差自编码器：

```powershell
.\.venv-enrc\Scripts\python.exe `
  experiments\pretrain.py `
  --dataset cmnist `
  --seed-base 42 `
  --n-aes 1 `
  --pretrain-steps 1000 `
  --finetune-steps 500 `
  --finetune-lr 0.0005 `
  --checkpoint-dir results\standard_enrc\cmnist\checkpoints
```

预训练完成后应生成：

```text
data/cmnist/models/ae-model-hl-16-idx-0.pth
```

运行 code-faithful ENRC：

```powershell
.\.venv-enrc\Scripts\python.exe `
  experiments\run_cmnist.py `
  --epochs 30 `
  --ae-lr 0.00025 `
  --beta-lr 0.001 `
  --device auto `
  --output-dir results\standard_enrc\cmnist
```

主要结果位于：

```text
results/standard_enrc/cmnist/code_faithful/metrics.json
```

## 4. NR-Objects

预训练：

```powershell
.\.venv-enrc\Scripts\python.exe `
  experiments\pretrain.py `
  --dataset nr_objects `
  --seed-base 42 `
  --n-aes 1 `
  --pretrain-steps 1000 `
  --finetune-steps 500 `
  --finetune-lr 0.0005 `
  --checkpoint-dir results\standard_enrc\nr_objects\checkpoints
```

运行 ENRC：

```powershell
.\.venv-enrc\Scripts\python.exe `
  experiments\run_nrobjects.py `
  --epochs 30 `
  --init-rounds 100 `
  --ae-lr 0.00025 `
  --beta-lr 0.001 `
  --device auto `
  --output-dir results\standard_enrc\nr_objects
```

## 5. 注意事项

- `--device auto` 优先使用 CUDA，否则使用 CPU。
- CPU 可以运行，但完整图像实验耗时较长。
- 当前命令复现单 seed 42 的参考实验，不是原作者十个自编码器的均值。
- 历史 `cluster_*.py` 文件仅用于对照原始源码；现代环境请使用
  `experiments/` 中的入口。
- Stickfigures 的统一 code-faithful 运行入口尚未加入。
- 原作者的 MIT License、原始 README 和原始 commit 信息均已保留。
