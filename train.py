"""Run from workspace root: python -m coral_auv_surrogate.train --help."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from .data import load_data
from .model import DragMLP, load_branch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--splits", help="NPZ with train/validation/test row indices; use grouped splits for correlated CFD")
    p.add_argument("--branch", choices=["translation", "rotation"], required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--loader", help="Custom CFD reader module:function")
    p.add_argument("--frame", required=True, help="Body-axis convention, e.g. x-forward_y-left_z-up")
    p.add_argument("--moment-reference", required=True, help="CFD moment origin, e.g. CAD_origin")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--normalize", action="store_true", help="Optional train-only input/output standardization; changes raw MSE weighting")
    args = p.parse_args()
    if args.epochs < 1 or args.hidden < 1 or (args.batch_size is not None and args.batch_size < 1):
        p.error("epochs, hidden and batch-size must be positive")
    if args.lr is not None and args.lr <= 0:
        p.error("lr must be positive")
    torch.manual_seed(args.seed)
    x, y = load_data(args.data, args.branch, args.loader)
    indices = np.random.default_rng(args.seed).permutation(len(x))
    a, b = int(.8 * len(x)), int(.9 * len(x))
    train_idx, val_idx, test_idx = indices[:a], indices[a:b], indices[b:]
    if args.splits:
        with np.load(args.splits, allow_pickle=False) as split:
            train_idx, val_idx, test_idx = [split[k] for k in ("train", "validation", "test")]
        all_indices = np.concatenate([train_idx, val_idx, test_idx])
        if (any(i.ndim != 1 or len(i) == 0 for i in (train_idx, val_idx, test_idx))
                or not np.issubdtype(all_indices.dtype, np.integer)
                or len(np.unique(all_indices)) != len(all_indices)
                or np.any(all_indices < 0) or np.any(all_indices >= len(x))):
            raise ValueError("Splits must be nonempty, disjoint integer row indices within the dataset")
    x, y = torch.from_numpy(x), torch.from_numpy(y)
    model = DragMLP(args.hidden, args.normalize).to(args.device)
    with torch.no_grad():
        model.velocity_min.copy_(x[train_idx].amin(0))
        model.velocity_max.copy_(x[train_idx].amax(0))
        if args.normalize:
            for name, values in [("x", x[train_idx]), ("y", y[train_idx])]:
                getattr(model, name + "_mean").copy_(values.mean(0))
                getattr(model, name + "_scale").copy_(values.std(0, unbiased=False).clamp_min(1e-6))
    batch = args.batch_size or (64 if args.branch == "translation" else 32)
    lr = args.lr or (2e-3 if args.branch == "translation" else 2e-4)
    batches = DataLoader(TensorDataset(x[train_idx], y[train_idx]), batch_size=batch, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "split_indices.npz", train=train_idx, validation=val_idx, test=test_idx)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    best = float("inf")
    with (out / "history.jsonl").open("w") as log:
        for epoch in range(1, args.epochs + 1):
            model.train()
            total = 0.0
            for xb, yb in batches:
                xb, yb = xb.to(args.device), yb.to(args.device)
                target = (yb - model.y_mean) / model.y_scale
                loss = torch.nn.functional.mse_loss(model.normalized_prediction(xb), target)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss; inspect CFD data and scaling")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item() * len(xb)
            model.eval()
            with torch.no_grad():
                vx, vy = x[val_idx].to(args.device), y[val_idx].to(args.device)
                val = torch.nn.functional.mse_loss(model.normalized_prediction(vx), (vy-model.y_mean)/model.y_scale).item()
            record = {"epoch": epoch, "train_loss": total/len(train_idx), "validation_loss": val}
            log.write(json.dumps(record) + "\n")
            log.flush()
            if val < best:
                best = val
                torch.save({"format_version": 1, "state_dict": model.state_dict(), "hidden": args.hidden,
                            "normalize": args.normalize, "branch": args.branch, "frame": args.frame,
                            "moment_reference": args.moment_reference, "units": "m/s,rad/s,N,N*m",
                            "epoch": epoch, "validation_loss": val}, out / "best.pt")
            if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
                print(json.dumps(record), flush=True)
    model, _ = load_branch(out / "best.pt", args.device)
    with torch.inference_mode():
        errors = (model(x[test_idx].to(args.device)) - y[test_idx].to(args.device)).cpu()
    metrics = {"columns": ["Fx", "Fy", "Fz", "Mx", "My", "Mz"],
               "test_samples": len(test_idx), "rmse_per_axis": errors.square().mean(0).sqrt().tolist(),
               "mae_per_axis": errors.abs().mean(0).tolist()}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
