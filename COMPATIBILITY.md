# Code-faithful ENRC compatibility port

This directory is based on the ENRC author release at commit
`6de4f30d38f82a609e7b58c0b62b822750870111`:

https://gitlab.cs.univie.ac.at/lukas/enrcpublic

The original source targeted Python 3.6, PyTorch 1.1.0, FastAI 1.0.54 and
Ubuntu 18.04.2. This copy retains the released model and clustering behavior
and applies only compatibility changes needed by the tested modern stack.

Applied source-level changes:

- replaced removed `numpy.int` aliases;
- replaced scikit-learn's removed private k-means++ entry point with the
  public `kmeans_plusplus` function;
- converted NumPy scalar seeds to Python integers;
- made the pretraining fine-tune helper follow the model device.

The experiment entry points additionally patch FastAI v1's removed
`torch.solve` dependency, force single-process data loading on Windows, and
replace incompatible legacy optimizer wrappers with explicit PyTorch Adam and
OneCycleLR loops.

No raw datasets, pretrained models, checkpoints or result files are included.
See `README_CN.md` in the repository root for setup and execution instructions.

This code remains subject to the original MIT License in this directory.
