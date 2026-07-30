"""Compare a spline edge-function surrogate with an MLP on two toy tasks.

The smooth task predicts the next hidden Hopf-fiber phase. The discrete task
predicts stale-cell conflicts from trace rows emitted by the exploratory Hex
simulator. The spline model is KAN-inspired, not a full KAN implementation.
"""

from __future__ import annotations

import csv
import hashlib
import math
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from run_hex_hopf_delay_simulation import DELAYS, OUT, POLICIES, simulate_game


SEED = 2026073300
DEVICE = torch.device("cpu")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def spline_features(x: np.ndarray, knots: int = 14) -> np.ndarray:
    """Additive triangular edge basis, a transparent KAN-like surrogate."""
    centers = np.linspace(-1.0, 1.0, knots)
    width = centers[1] - centers[0]
    blocks = [np.maximum(1.0 - np.abs(x[:, j, None] - centers[None, :]) / width, 0.0) for j in range(x.shape[1])]
    return np.concatenate([np.ones((x.shape[0], 1)), *blocks], axis=1)


def ridge_fit(x: np.ndarray, y: np.ndarray, regularization: float = 1e-5) -> np.ndarray:
    gram = x.T @ x + regularization * np.eye(x.shape[1])
    return np.linalg.solve(gram, x.T @ y)


class MLP(nn.Module):
    def __init__(self, inputs: int, outputs: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(inputs, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, outputs))

    def forward(self, x):
        return self.net(x)


def train_mlp(x_train, y_train, outputs: int, classification: bool):
    torch.manual_seed(SEED)
    model = MLP(x_train.shape[1], outputs).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss() if classification else nn.MSELoss()
    x_tensor = torch.tensor(x_train, dtype=torch.float32, device=DEVICE)
    y_tensor = torch.tensor(y_train, dtype=torch.float32, device=DEVICE)
    start = time.perf_counter()
    for _ in range(250 if classification else 350):
        optimizer.zero_grad()
        loss = criterion(model(x_tensor), y_tensor)
        loss.backward()
        optimizer.step()
    return model, time.perf_counter() - start


def smooth_dataset(n: int = 5000):
    rng = np.random.default_rng(SEED)
    phi = rng.uniform(-math.pi, math.pi, n)
    neighbor = rng.uniform(-math.pi, math.pi, n)
    omega = rng.uniform(-0.08, 0.08, n)
    delay = rng.uniform(0.0, 5.0, n)
    next_phi = phi + omega + 0.18 * np.sin(neighbor - phi) + 0.015 * delay
    x = np.column_stack((np.sin(phi), np.cos(phi), np.sin(neighbor), np.cos(neighbor), omega / 0.08, delay / 5.0))
    y = np.column_stack((np.sin(next_phi), np.cos(next_phi)))
    return x, y


def conflict_dataset(games_per_cell: int = 8):
    rows = []
    for delay in DELAYS:
        for policy_index, policy in enumerate(POLICIES):
            for index in range(games_per_cell):
                result = simulate_game(delay, policy, SEED + delay * 10000 + policy_index * 1000 + index, keep_trace=True)
                for item in result["trace"]:
                    if item["selected_phase_error"] is None:
                        continue
                    rows.append((
                        delay / 5.0,
                        float(policy_index),
                        min(item["pending_count"] / 50.0, 1.0),
                        item["selected_phase_error"] / math.pi,
                        min(item["turn"] / 250.0, 1.0),
                        float(item["stale_conflict"]),
                    ))
    data = np.asarray(rows, dtype=float)
    return data[:, :-1], data[:, -1:]


def circular_mae(pred_sincos, target_sincos):
    pred = np.arctan2(pred_sincos[:, 0], pred_sincos[:, 1])
    target = np.arctan2(target_sincos[:, 0], target_sincos[:, 1])
    return float(np.mean(np.abs(np.arctan2(np.sin(pred - target), np.cos(pred - target)))))


def evaluate():
    rng = np.random.default_rng(SEED)
    metrics = []

    smooth_x, smooth_y = smooth_dataset()
    split = int(0.8 * len(smooth_x))
    train_x, test_x = smooth_x[:split], smooth_x[split:]
    train_y, test_y = smooth_y[:split], smooth_y[split:]
    start = time.perf_counter()
    spline_weights = ridge_fit(spline_features(train_x * 2.0 - 1.0), train_y)
    spline_time = time.perf_counter() - start
    spline_pred = spline_features(test_x * 2.0 - 1.0) @ spline_weights
    metrics.append({"task": "smooth_hopf_phase", "model": "spline_edge_surrogate", "metric": "circular_mae_radians", "value": circular_mae(spline_pred, test_y), "fit_seconds": spline_time})
    mlp, mlp_time = train_mlp(train_x, train_y, 2, False)
    with torch.no_grad():
        mlp_pred = mlp(torch.tensor(test_x, dtype=torch.float32)).cpu().numpy()
    metrics.append({"task": "smooth_hopf_phase", "model": "mlp", "metric": "circular_mae_radians", "value": circular_mae(mlp_pred, test_y), "fit_seconds": mlp_time})

    conflict_x, conflict_y = conflict_dataset()
    order = rng.permutation(len(conflict_x))
    split = int(0.8 * len(order))
    train_idx, test_idx = order[:split], order[split:]
    train_x, test_x = conflict_x[train_idx], conflict_x[test_idx]
    train_y, test_y = conflict_y[train_idx], conflict_y[test_idx]
    spline_weights = ridge_fit(spline_features(train_x * 2.0 - 1.0), train_y)
    spline_pred = (spline_features(test_x * 2.0 - 1.0) @ spline_weights).ravel() >= 0.5
    true = test_y.ravel() >= 0.5
    metrics.append({"task": "discrete_stale_conflict", "model": "spline_edge_surrogate", "metric": "accuracy", "value": float(np.mean(spline_pred == true)), "fit_seconds": 0.0})
    metrics.append({"task": "discrete_stale_conflict", "model": "spline_edge_surrogate", "metric": "positive_recall", "value": float(np.sum(spline_pred & true) / max(1, np.sum(true))), "fit_seconds": 0.0})
    mlp, mlp_time = train_mlp(train_x, train_y, 1, True)
    with torch.no_grad():
        mlp_pred = torch.sigmoid(mlp(torch.tensor(test_x, dtype=torch.float32))).cpu().numpy().ravel() >= 0.5
    metrics.append({"task": "discrete_stale_conflict", "model": "mlp", "metric": "accuracy", "value": float(np.mean(mlp_pred == true)), "fit_seconds": mlp_time})
    metrics.append({"task": "discrete_stale_conflict", "model": "mlp", "metric": "positive_recall", "value": float(np.sum(mlp_pred & true) / max(1, np.sum(true))), "fit_seconds": mlp_time})
    return metrics, len(conflict_x)


def main():
    metrics, conflict_rows = evaluate()
    csv_path = OUT / "hopf_model_selection_metrics.csv"
    write_csv(csv_path, metrics)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.9))
    colors = {"spline_edge_surrogate": "#D55E00", "mlp": "#0072B2"}
    for axis, task, metric, title, ylabel in [
        (axes[0], "smooth_hopf_phase", "circular_mae_radians", "Smooth Hopf phase", "Circular MAE (radians)"),
        (axes[1], "discrete_stale_conflict", "accuracy", "Discrete stale conflict", "Accuracy"),
    ]:
        selected = [row for row in metrics if row["task"] == task and row["metric"] == metric]
        labels = ["spline edge" if row["model"] == "spline_edge_surrogate" else "MLP" for row in selected]
        axis.bar(labels, [row["value"] for row in selected], color=[colors[row["model"]] for row in selected])
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=0)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Edge-function model selection under delayed Hopf/Hex toy dynamics", fontsize=14, fontweight="bold")
    fig.text(0.01, 0.01, f"Conflict evaluation rows: {conflict_rows}; spline is a KAN-inspired surrogate, not a full KAN; exploratory only.", fontsize=8)
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    figure = OUT / "hopf_model_selection_metrics.png"
    fig.savefig(figure, dpi=220, facecolor="white")
    plt.close(fig)

    report = OUT / "hopf_model_selection_report.md"
    report.write_text("# Hopf/Hex Model Selection Check\n\n" + "The smooth task predicts the next hidden Hopf-fiber phase. The discrete task predicts stale-cell conflicts from retained exploratory Hex trace rows. The spline model is an additive triangular edge-function surrogate inspired by the KAN parameterization; it is not presented as a full KAN implementation.\n\n" + "| task | model | metric | value | fit seconds |\n|---|---|---|---:|---:|\n" + "\n".join(f"| {row['task']} | {row['model']} | {row['metric']} | {float(row['value']):.17g} | {float(row['fit_seconds']):.17g} |" for row in metrics) + "\n\nThe result is a model-selection diagnostic, not evidence that KANs improve runtime assurance. It tests the paper's central practical caution: a smooth function-space representation can be useful on a smooth phase task while offering no automatic solution to discrete queue conflicts.\n", encoding="utf-8")
    manifest = OUT / "HOPF_MODEL_SELECTION_MANIFEST.sha256"
    sources = [Path(__file__), Path(__file__).with_name("run_hex_hopf_delay_simulation.py"), csv_path, figure, report]
    manifest.write_text("\n".join(f"{sha256(path)} *{path}" for path in sources) + "\n", encoding="utf-8")
    print(f"Metrics: {csv_path}")
    print(f"Figure: {figure}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
