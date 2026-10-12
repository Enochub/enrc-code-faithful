from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ENRCPUBLIC = ROOT
for path in (HERE, ENRCPUBLIC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from enrc.enrc import Enrc  # noqa: E402


class ModelCaptured(RuntimeError):
    pass


def build_loaded_cmnist_model():
    import fastai.vision.transform as vision_transform
    from fastai.basic_train import Learner
    from fastai.data_block import LabelLists

    vision_transform._solve_func = lambda b, a: (torch.linalg.solve(a, b), None)
    original_databunch = LabelLists.databunch

    def cpu_databunch(self, *args, **kwargs):
        kwargs["num_workers"] = 0
        return original_databunch(self, *args, **kwargs)

    LabelLists.databunch = cpu_databunch
    captured = {}

    def load_and_capture(self, *args, **kwargs):
        del args, kwargs
        self.load("ae-model-hl-16-idx-0", device=torch.device("cpu"))
        captured["learner"] = self
        raise ModelCaptured

    Learner.fit_one_cycle = load_and_capture
    import pretrain_cmnist

    try:
        pretrain_cmnist.pretrain(np.arange(42, 52, dtype=np.int64))
    except ModelCaptured:
        pass
    learner = captured["learner"]
    learner.model.cpu().eval()
    return learner.model, learner.data


def materialize(data):
    batches = []
    for batch in data.fix_dl:
        batches.append(batch[0].cpu())
    x = torch.cat(batches)
    paths = list(data.fix_dl.dataset.x.items)
    labels = np.asarray([[int(Path(p).parent.name.split("_")[0]),
                          int(Path(p).parent.name.split("_")[1])] for p in paths])
    if len(x) != len(labels):
        raise RuntimeError(f"data/label length mismatch: {len(x)} != {len(labels)}")
    return x, labels


def clustering_accuracy(y_true, y_pred):
    true_values = np.unique(y_true)
    pred_values = np.unique(y_pred)
    matrix = np.zeros((len(pred_values), len(true_values)), dtype=np.int64)
    for i, pred in enumerate(pred_values):
        for j, true in enumerate(true_values):
            matrix[i, j] = np.sum((y_pred == pred) & (y_true == true))
    rows, cols = linear_sum_assignment(-matrix)
    return float(matrix[rows, cols].sum() / len(y_true))


def matched_metrics(labels, predictions):
    best = None
    for permutation in itertools.permutations(range(predictions.shape[1]), labels.shape[1]):
        rows = []
        for truth_id, prediction_id in enumerate(permutation):
            truth = labels[:, truth_id]
            prediction = predictions[:, prediction_id]
            rows.append({
                "truth_space": truth_id,
                "predicted_space": prediction_id,
                "acc": clustering_accuracy(truth, prediction),
                "nmi": float(normalized_mutual_info_score(truth, prediction)),
                "ari": float(adjusted_rand_score(truth, prediction)),
            })
        score = sum(row["nmi"] for row in rows)
        if best is None or score > best[0]:
            best = (score, rows)
    return best[1]


def run_variant(name, layer_type, model, base_state, data, x, labels, epochs, output_dir,
                diagnose_joint=False, diagnostic_steps=100,
                diagnostic_lrs=(0.0025, 0.0005, 0.00025, 0.0001),
                formal_ae_lr=0.0025, formal_beta_lr=0.01,
                device=torch.device("cpu")):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    model.load_state_dict(copy.deepcopy(base_state))
    model.to(device).eval()
    batch_size = data.train_dl.batch_size

    def embedding_fn(values):
        encoded = []
        with torch.no_grad():
            for start in range(0, len(values), batch_size):
                encoded.append(model.encode(values[start:start + batch_size].to(device)).cpu())
        return torch.cat(encoded)

    print(f"{name} INIT_START samples={len(x)} rounds=100", flush=True)
    layer = layer_type.run_init(x, [10, 10], embedding_fn, rounds=100,
                                random_state=42, max_iter=10)
    layer.to(device)
    layer.centers = [center.to(device) for center in layer.centers]
    layer.mask_sum = [value.to(device) for value in layer.mask_sum]
    layer.center_lr = 0.5
    initial_layer_state = copy.deepcopy(layer.state_dict())
    initial_centers = [center.clone() for center in layer.centers]
    initial_mask_sum = [value.clone() for value in layer.mask_sum]
    initial_lonely = [value.copy() for value in layer.lonely_centers_count]
    init_dir = output_dir / name
    init_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": base_state,
        "layer_state": initial_layer_state,
        "centers": initial_centers,
        "mask_sum": initial_mask_sum,
        "lonely_centers_count": initial_lonely,
        "init_P": layer.init_P,
        "center_lr": layer.center_lr,
    }, init_dir / "initialization.pth")
    print(f"{name} INIT_SAVED path={init_dir / 'initialization.pth'}", flush=True)

    if diagnose_joint:
        parameters = list(model.parameters()) + [layer.rotation_layer.v, layer.beta_weights]
        for ae_lr in diagnostic_lrs:
            model.load_state_dict(copy.deepcopy(base_state))
            layer.load_state_dict(copy.deepcopy(initial_layer_state))
            layer.centers = [center.clone() for center in initial_centers]
            layer.mask_sum = [value.clone() for value in initial_mask_sum]
            layer.lonely_centers_count = [value.copy() for value in initial_lonely]
            layer.reinit_threshold = 1
            layer.center_lr = 0.5
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
            beta_lr = ae_lr * 4.0
            optimizer = torch.optim.Adam([
                {"params": model.parameters(), "lr": ae_lr},
                {"params": [layer.rotation_layer.v], "lr": ae_lr},
                {"params": [layer.beta_weights], "lr": beta_lr},
            ])
            iterator = iter(data.train_dl)
            print(f"JOINT_DIAGNOSTIC_START ae_lr={ae_lr:.8g} beta_lr={beta_lr:.8g}", flush=True)
            for diagnostic_step in range(1, diagnostic_steps + 1):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(data.train_dl)
                    batch = next(iterator)
                model.eval()
                values = batch[0].to(device)
                z = model.encode(values)
                cluster_loss, z_back = layer(z)
                reconstruction_loss = torch.nn.functional.mse_loss(model.decode(z_back), values)
                loss = cluster_loss + reconstruction_loss
                optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, float("inf"))
                if not torch.isfinite(loss) or not torch.isfinite(grad_norm):
                    print(f"JOINT_DIAGNOSTIC_DIVERGED ae_lr={ae_lr:.8g} "
                          f"step={diagnostic_step} loss={loss.item()} grad_norm={float(grad_norm)}",
                          flush=True)
                    break
                optimizer.step()
                for subspace in range(len(layer.centers)):
                    layer.reinit_centers(subspace, n_samples=1000, kmeans_steps=10)
                layer.reinit_threshold = int(np.sqrt(diagnostic_step))
                if diagnostic_step == 1 or diagnostic_step % 10 == 0:
                    parameter_max = max(value.detach().abs().max().item() for value in parameters)
                    print(f"JOINT_DIAGNOSTIC ae_lr={ae_lr:.8g} step={diagnostic_step}/{diagnostic_steps} "
                          f"loss={loss.item():.8g} cluster={cluster_loss.item():.8g} "
                          f"rec={reconstruction_loss.item():.8g} grad_norm={float(grad_norm):.8g} "
                          f"parameter_max={parameter_max:.8g}", flush=True)
        return

    optimizer = torch.optim.Adam([
        {"params": model.parameters(), "lr": formal_ae_lr},
        {"params": [layer.rotation_layer.v], "lr": formal_ae_lr},
        {"params": [layer.beta_weights], "lr": formal_beta_lr},
    ])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)
    step = 0
    for epoch in range(1, epochs + 1):
        for batch in data.train_dl:
            model.eval()
            values = batch[0].to(device)
            z = model.encode(values)
            cluster_loss, z_back = layer(z)
            reconstruction_loss = torch.nn.functional.mse_loss(model.decode(z_back), values)
            loss = cluster_loss + reconstruction_loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"{name}: non-finite loss at epoch={epoch}, step={step}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            for subspace in range(len(layer.centers)):
                layer.reinit_centers(subspace, n_samples=1000, kmeans_steps=10)
            step += 1
            layer.reinit_threshold = int(np.sqrt(step))
            scheduler.step()
            if step % 2000 == 0:
                layer.center_lr *= 0.5
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"{name} epoch={epoch}/{epochs} step={step} loss={loss.item():.8g} "
                  f"cluster={cluster_loss.item():.8g} rec={reconstruction_loss.item():.8g}", flush=True)

    model.eval()
    predicted = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            predicted.append(layer.predict(model.encode(x[start:start + batch_size].to(device))))
    predictions = np.concatenate(predicted)
    metrics = matched_metrics(labels, predictions)
    result = {"variant": name, "epochs": epochs, "steps": step, "metrics": metrics,
              "mean_acc": float(np.mean([row["acc"] for row in metrics])),
              "mean_nmi": float(np.mean([row["nmi"] for row in metrics])),
              "mean_ari": float(np.mean([row["ari"] for row in metrics]))}
    variant_dir = output_dir / name
    variant_dir.mkdir(parents=True, exist_ok=True)
    np.save(variant_dir / "predictions.npy", predictions)
    torch.save({"model": model.state_dict(), "layer": layer.state_dict()}, variant_dir / "joint_model.pth")
    (variant_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"{name} RESULT {json.dumps(result)}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--diagnose-joint", action="store_true")
    parser.add_argument("--diagnostic-steps", type=int, default=100)
    parser.add_argument("--diagnostic-lrs", default="0.0025,0.0005,0.00025,0.0001")
    parser.add_argument("--ae-lr", type=float, default=0.0025)
    parser.add_argument("--beta-lr", type=float, default=0.01)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    os.chdir(ENRCPUBLIC)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    print(f"RUN_DEVICE={device}", flush=True)
    model, data = build_loaded_cmnist_model()
    base_state = copy.deepcopy(model.state_dict())
    x, labels = materialize(data)
    print(f"CMNIST_READY x={tuple(x.shape)} labels={tuple(labels.shape)}", flush=True)
    diagnostic_lrs = tuple(float(value) for value in args.diagnostic_lrs.split(","))
    run_variant("code_faithful", Enrc, model, base_state, data, x, labels, args.epochs,
                args.output_dir, args.diagnose_joint, args.diagnostic_steps, diagnostic_lrs,
                args.ae_lr, args.beta_lr, device)


if __name__ == "__main__":
    main()
