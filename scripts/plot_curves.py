# -*- coding: utf-8 -*-
"""
OralGuard -- Training Curve Plotter
Reads metrics from the MLflow SQLite database for the most recent
FINISHED run and saves loss_curve.png + f1_curve.png to assets/.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless backend — no GUI needed
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT      = Path(__file__).resolve().parent.parent
DB_PATH   = ROOT / "mlflow" / "mlflow.db"
ASSETS    = ROOT / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# House-style colours (dark mode palette)
# ---------------------------------------------------------------------------
BG        = "#0d1117"
PANEL     = "#161b22"
GRID      = "#21262d"
ACCENT1   = "#58a6ff"   # blue  → train_loss
ACCENT2   = "#f78166"   # coral → val_loss
ACCENT3   = "#3fb950"   # green → val_f1
STAR      = "#ffa657"   # orange marker → best epoch
TEXT      = "#e6edf3"
MUTED     = "#8b949e"

# ---------------------------------------------------------------------------
# Fetch metrics
# ---------------------------------------------------------------------------

def fetch_run(conn: sqlite3.Connection) -> tuple[str, pd.DataFrame]:
    """Return (run_name, metrics_df) for the most-recently FINISHED run."""
    runs = conn.execute(
        "SELECT run_uuid, name FROM runs "
        "WHERE status = 'FINISHED' ORDER BY start_time DESC LIMIT 1"
    ).fetchall()
    if not runs:
        raise RuntimeError("No FINISHED runs found in MLflow database.")

    run_uuid, run_name = runs[0]
    df = pd.read_sql_query(
        f"SELECT step, key, value FROM metrics "
        f"WHERE run_uuid = '{run_uuid}' AND step > 0 "
        f"ORDER BY step, key",
        conn,
    )
    if df.empty:
        raise RuntimeError(f"No epoch metrics found for run {run_name}.")

    pivoted = df.pivot_table(index="step", columns="key", values="value")
    return run_name, pivoted


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _base_fig(nrows: int = 1) -> tuple[plt.Figure, list]:
    """Create a figure with dark background and return (fig, axes)."""
    fig, axes = plt.subplots(
        nrows, 1,
        figsize=(10, 4 * nrows),
        facecolor=BG,
    )
    if nrows == 1:
        axes = [axes]
    for ax in axes:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=MUTED, labelsize=10)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)
        ax.spines[:].set_edgecolor(GRID)
        ax.grid(True, color=GRID, linewidth=0.8, linestyle="--", alpha=0.7)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    return fig, axes


def _mark_best(ax: plt.Axes, x: int, y: float, label: str) -> None:
    """Drop a star marker at the best epoch."""
    ax.axvline(x=x, color=STAR, linewidth=1.0, linestyle=":", alpha=0.7)
    ax.scatter([x], [y], color=STAR, zorder=5, s=90, marker="*", label=label)


# ---------------------------------------------------------------------------
# loss_curve.png
# ---------------------------------------------------------------------------

def plot_loss(df: pd.DataFrame, run_name: str, out_path: Path) -> None:
    fig, (ax,) = _base_fig(1)

    epochs = df.index.to_numpy()
    train  = df["train_loss"].to_numpy()
    val    = df["val_loss"].to_numpy()

    ax.plot(epochs, train, color=ACCENT1, linewidth=2.0, label="Train Loss")
    ax.plot(epochs, val,   color=ACCENT2, linewidth=2.0, label="Val Loss")

    best_ep = int(df["val_loss"].idxmin())
    best_vl = float(df.loc[best_ep, "val_loss"])
    _mark_best(ax, best_ep, best_vl, f"Best Val Loss = {best_vl:.4f} @ ep {best_ep}")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("BCE Loss (weighted)", fontsize=12)
    ax.set_title(f"OralGuard Classifier — Training & Validation Loss\n{run_name}",
                 fontsize=13, fontweight="bold", pad=12)

    legend = ax.legend(
        facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT,
        fontsize=10, loc="upper right",
    )

    fig.tight_layout(pad=2.0)
    fig.savefig(out_path, dpi=180, facecolor=BG)
    plt.close(fig)
    print(f"  ✅  Saved: {out_path}")


# ---------------------------------------------------------------------------
# f1_curve.png
# ---------------------------------------------------------------------------

def plot_f1(df: pd.DataFrame, run_name: str, out_path: Path) -> None:
    fig, (ax,) = _base_fig(1)

    epochs = df.index.to_numpy()
    f1     = df["val_f1"].to_numpy()

    ax.fill_between(epochs, 0, f1, color=ACCENT3, alpha=0.15)
    ax.plot(epochs, f1, color=ACCENT3, linewidth=2.0, label="Val F1 (macro)")

    best_ep = int(df["val_f1"].idxmax())
    best_f1 = float(df.loc[best_ep, "val_f1"])
    _mark_best(ax, best_ep, best_f1, f"Best Val F1 = {best_f1:.4f} @ ep {best_ep}")

    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Macro F1 Score", fontsize=12)
    ax.set_title(f"OralGuard Classifier — Validation F1 Score\n{run_name}",
                 fontsize=13, fontweight="bold", pad=12)

    legend = ax.legend(
        facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT,
        fontsize=10, loc="lower right",
    )

    fig.tight_layout(pad=2.0)
    fig.savefig(out_path, dpi=180, facecolor=BG)
    plt.close(fig)
    print(f"  ✅  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Reading MLflow database: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    run_name, df = fetch_run(conn)
    conn.close()

    print(f"Using run: {run_name}  ({len(df)} epochs)")
    print(f"Columns  : {list(df.columns)}")

    plot_loss(df, run_name, ASSETS / "loss_curve.png")
    plot_f1  (df, run_name, ASSETS / "f1_curve.png")

    print("\nDone! Both charts saved to assets/")


if __name__ == "__main__":
    main()
