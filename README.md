# ENRC code-faithful compatibility source

This repository contains the ENRC author implementation at upstream commit
`6de4f30d38f82a609e7b58c0b62b822750870111`, with the minimal source-level
compatibility changes documented in [`COMPATIBILITY.md`](COMPATIBILITY.md).

Do not use the historical `cluster_*.py` files as public entry points in a
modern environment. They are retained for comparison with the upstream
release. Use the tested wrappers instead:

```text
experiments/pretrain.py
experiments/run_cmnist.py
experiments/run_nrobjects.py
```

Chinese setup and execution instructions are available in:

```text
README_CN.md
```

The unmodified upstream README is preserved as
[`UPSTREAM_README.md`](UPSTREAM_README.md). The original MIT License is
preserved in this directory.
