from __future__ import annotations

import argparse
import copy
import importlib
import os
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
ENRCPUBLIC = ROOT
if str(ENRCPUBLIC) not in sys.path:
    sys.path.insert(0, str(ENRCPUBLIC))


class RequestedRunsComplete(RuntimeError):
    pass


def install_fastai_compatibility(
    n_aes: int, pretrain_steps: int, load_pretrain_checkpoint: Path | None = None
) -> None:
    import fastai.vision.transform as vision_transform
    from fastai.basic_train import Learner
    from fastai.data_block import LabelLists

    # fastai v1 calls the removed torch.solve API in geometric augmentation.
    vision_transform._solve_func = lambda b, a: (torch.linalg.solve(a, b), None)

    # Worker processes would import the unpatched fastai module on Windows.
    original_databunch = LabelLists.databunch

    def single_process_databunch(self, *args, **kwargs):
        kwargs["num_workers"] = 0
        return original_databunch(self, *args, **kwargs)

    LabelLists.databunch = single_process_databunch

    def exact_one_cycle(self, cyc_len, max_lr=None, wd=None, **kwargs):
        del cyc_len, kwargs
        if load_pretrain_checkpoint is not None:
            checkpoint = torch.load(load_pretrain_checkpoint, map_location="cpu")
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(
                f"PRETRAIN_CHECKPOINT_LOADED path={load_pretrain_checkpoint}",
                flush=True,
            )
            return
        total_steps = pretrain_steps
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=max_lr, betas=(0.9, 0.99), weight_decay=wd
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lr, total_steps=total_steps,
            pct_start=0.5, anneal_strategy="cos", cycle_momentum=False,
        )
        self.model.train()
        iterator = iter(self.data.train_dl)
        for step in range(total_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(self.data.train_dl)
                batch = next(iterator)
            inputs, targets = batch[0], batch[1]
            predictions = self.model(inputs)
            loss = self.loss_func(predictions, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            if step == 0 or (step + 1) % 100 == 0:
                print(
                    f"PRETRAIN step={step + 1}/{total_steps} "
                    f"loss={loss.item():.8f} lr={scheduler.get_last_lr()[0]:.8g}",
                    flush=True,
                )

    Learner.fit_one_cycle = exact_one_cycle

    original_save = Learner.save
    saved = {"count": 0}

    def save_requested_runs(self, *args, **kwargs):
        result = original_save(self, *args, **kwargs)
        saved["count"] += 1
        print(f"CHECKPOINT_SAVED count={saved['count']}/{n_aes}", flush=True)
        if saved["count"] >= n_aes:
            raise RequestedRunsComplete(f"Completed requested {n_aes} AE(s)")
        return result

    Learner.save = save_requested_runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cmnist", "nr_objects", "stickfigures"], required=True)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--n-aes", type=int, default=1)
    parser.add_argument("--pretrain-steps", type=int, default=1_000)
    parser.add_argument("--finetune-steps", type=int, default=500)
    parser.add_argument("--finetune-lr", type=float)
    parser.add_argument("--load-pretrain-checkpoint", type=Path)
    parser.add_argument("--diagnose-finetune", action="store_true")
    parser.add_argument("--diagnostic-steps", type=int, default=100)
    parser.add_argument(
        "--diagnostic-lrs", default="0.005,0.001,0.0005,0.0001"
    )
    parser.add_argument("--checkpoint-dir", type=Path)
    args = parser.parse_args()
    if args.checkpoint_dir is not None:
        args.checkpoint_dir = args.checkpoint_dir.resolve()
    if args.load_pretrain_checkpoint is not None:
        args.load_pretrain_checkpoint = args.load_pretrain_checkpoint.resolve()
    os.chdir(ENRCPUBLIC)
    install_fastai_compatibility(
        args.n_aes, args.pretrain_steps, args.load_pretrain_checkpoint
    )
    module_names = {
        "cmnist": "pretrain_cmnist",
        "nr_objects": "pretrain_nrobjects",
        "stickfigures": "pretrain_stickfigures",
    }
    module = importlib.import_module(module_names[args.dataset])
    seeds = np.arange(args.seed_base, args.seed_base + 10, dtype=np.int64)
    def requested_finetune(learn, n_iterations, lr):
        del n_iterations
        if args.finetune_lr is not None:
            lr = args.finetune_lr
        if args.checkpoint_dir is not None:
            args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            clean_path = args.checkpoint_dir / (
                f"{args.dataset}_seed{args.seed_base}_pretrain{args.pretrain_steps}.pth"
            )
            clean_state = copy.deepcopy(learn.model.state_dict())
            torch.save(
                {
                    "dataset": args.dataset,
                    "seed": args.seed_base,
                    "pretrain_steps": args.pretrain_steps,
                    "model_state_dict": clean_state,
                },
                clean_path,
            )
            print(f"PRETRAIN_CHECKPOINT_SAVED path={clean_path}", flush=True)
        else:
            clean_state = copy.deepcopy(learn.model.state_dict())

        if args.diagnose_finetune:
            learning_rates = [float(value) for value in args.diagnostic_lrs.split(",")]
            parameters = list(learn.model.parameters())
            device = parameters[0].device
            for diagnostic_lr in learning_rates:
                learn.model.load_state_dict(clean_state)
                learn.model.eval()
                torch.manual_seed(args.seed_base)
                np.random.seed(args.seed_base)
                optimizer = torch.optim.Adam(
                    learn.model.parameters(), lr=diagnostic_lr, betas=(0.9, 0.99)
                )
                iterator = iter(learn.data.train_dl)
                print(f"DIAGNOSTIC_START lr={diagnostic_lr:.8g}", flush=True)
                for step in range(args.diagnostic_steps):
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        iterator = iter(learn.data.train_dl)
                        batch = next(iterator)
                    inputs = batch[0].to(device)
                    reconstructed = learn.model.decode(learn.model.encode(inputs))
                    loss = torch.nn.functional.mse_loss(reconstructed, inputs)
                    optimizer.zero_grad()
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, float("inf"))
                    optimizer.step()
                    if step == 0 or (step + 1) % 10 == 0:
                        parameter_max = max(p.detach().abs().max().item() for p in parameters)
                        output_max = reconstructed.detach().abs().max().item()
                        print(
                            f"DIAGNOSTIC lr={diagnostic_lr:.8g} "
                            f"step={step + 1}/{args.diagnostic_steps} "
                            f"loss={loss.item():.8g} grad_norm={float(grad_norm):.8g} "
                            f"parameter_max={parameter_max:.8g} output_max={output_max:.8g}",
                            flush=True,
                        )
                if args.checkpoint_dir is not None:
                    lr_tag = format(diagnostic_lr, ".8g").replace(".", "p")
                    torch.save(
                        {
                            "dataset": args.dataset,
                            "seed": args.seed_base,
                            "pretrain_steps": args.pretrain_steps,
                            "finetune_steps": args.diagnostic_steps,
                            "finetune_lr": diagnostic_lr,
                            "model_state_dict": learn.model.state_dict(),
                        },
                        args.checkpoint_dir / f"{args.dataset}_seed{args.seed_base}_finetune_lr{lr_tag}.pth",
                    )
            raise RequestedRunsComplete("Fine-tune diagnostics complete")

        # The public helper delegates to fastai's optimizer factory. With the
        # installed legacy-fastai/PyTorch combination that produced an invalid,
        # exploding checkpoint. Spell out the paper's Adam update directly.
        learn.model.eval()
        optimizer = torch.optim.Adam(
            learn.model.parameters(), lr=lr, betas=(0.9, 0.99)
        )
        parameters = list(learn.model.parameters())
        iterator = iter(learn.data.train_dl)
        for step in range(args.finetune_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(learn.data.train_dl)
                batch = next(iterator)
            inputs = batch[0].to(next(learn.model.parameters()).device)
            reconstructed = learn.model.decode(learn.model.encode(inputs))
            loss = torch.nn.functional.mse_loss(reconstructed, inputs)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite fine-tune loss at step {step + 1}")
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, float("inf"))
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f"Non-finite gradient norm at step {step + 1}")
            optimizer.step()
            if step == 0 or (step + 1) % 50 == 0:
                parameter_max = max(p.detach().abs().max().item() for p in parameters)
                output_max = reconstructed.detach().abs().max().item()
                print(
                    f"FINETUNE step={step + 1}/{args.finetune_steps} "
                    f"lr={lr:.8g} loss={loss.item():.8f} "
                    f"grad_norm={float(grad_norm):.8g} "
                    f"parameter_max={parameter_max:.8g} output_max={output_max:.8g}",
                    flush=True,
                )

    module.finetune = requested_finetune
    print(f"DATASET={args.dataset} SEEDS={seeds.tolist()}", flush=True)
    try:
        module.pretrain(seeds)
    except RequestedRunsComplete as error:
        print(str(error), flush=True)


if __name__ == "__main__":
    main()
