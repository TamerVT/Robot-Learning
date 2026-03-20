"""Data quality inspection for collected teleop demonstrations.

Usage (raw session):
    python scripts/inspect_data.py --zarr datasets/backup/so100_transfer_cube_teleop.zarr

Usage (processed zarr, after compute_actions.py):
    python scripts/inspect_data.py --zarr datasets/processed/single_cube/processed_ee_xyz.zarr \
        --processed --state-keys state_ee_xyz state_gripper --action-keys action_ee_xyz action_gripper

Pass --no-plot to skip matplotlib figures (print stats only).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import zarr


# ── helpers ──────────────────────────────────────────────────────────────────


def load_raw_zarr(zarr_path: Path):
    """Load a raw teleop zarr (output of record_teleop_demos.py)."""
    root = zarr.open_group(str(zarr_path), mode="r")
    ep_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    data = root["data"]

    ee = np.asarray(data["state_ee"][:], dtype=np.float32) if "state_ee" in data else None
    gripper = np.asarray(data["state_gripper"][:], dtype=np.float32)
    cube = np.asarray(data["state_cube"][:], dtype=np.float32) if "state_cube" in data else None
    action_gripper = np.asarray(data["action_gripper"][:], dtype=np.float32) if "action_gripper" in data else None
    return ep_ends, ee, gripper, cube, action_gripper


def load_processed_zarr(zarr_path: Path, state_keys: list[str], action_keys: list[str]):
    """Load a processed zarr (output of compute_actions.py)."""
    root = zarr.open_group(str(zarr_path), mode="r")
    ep_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    data = root["data"]

    def _get(key):
        base = key.split("[")[0]
        if base not in data:
            print(f"  [warn] key '{base}' not found in zarr, skipping")
            return None
        return np.asarray(data[base][:], dtype=np.float32)

    states = {k.split("[")[0]: _get(k) for k in state_keys}
    actions = {k.split("[")[0]: _get(k) for k in action_keys}
    return ep_ends, states, actions


def episode_lengths(ep_ends: np.ndarray) -> np.ndarray:
    starts = np.concatenate(([0], ep_ends[:-1]))
    return ep_ends - starts


def idle_fraction(arr: np.ndarray, thresh: float = 1e-4) -> float:
    """Fraction of timesteps where the L2 norm of the row is below thresh (near-zero movement)."""
    norms = np.linalg.norm(arr, axis=-1) if arr.ndim > 1 else np.abs(arr.ravel())
    return float(np.mean(norms < thresh))


# ── printing helpers ──────────────────────────────────────────────────────────


def print_section(title: str):
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print('═' * 60)


def print_array_stats(name: str, arr: np.ndarray):
    if arr is None:
        return
    flat = arr.reshape(len(arr), -1)
    print(f"\n  {name}  shape={arr.shape}")
    for i in range(flat.shape[1]):
        col = flat[:, i]
        print(f"    dim[{i}]  min={col.min():.4f}  max={col.max():.4f}  "
              f"mean={col.mean():.4f}  std={col.std():.4f}")


# ── main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr", type=Path, required=True)
    parser.add_argument("--processed", action="store_true",
                        help="Set if the zarr is a processed (post-compute_actions) store.")
    parser.add_argument("--state-keys", nargs="+", default=["state_ee_xyz", "state_gripper"],
                        help="State keys to inspect (processed mode only).")
    parser.add_argument("--action-keys", nargs="+", default=["action_ee_xyz", "action_gripper"],
                        help="Action keys to inspect (processed mode only).")
    parser.add_argument("--no-plot", action="store_true", help="Skip matplotlib plots.")
    parser.add_argument("--idle-thresh", type=float, default=1e-4,
                        help="L2-norm threshold below which a timestep is considered idle.")
    args = parser.parse_args()

    # ── load ─────────────────────────────────────────────────────────
    if args.processed:
        ep_ends, states, actions = load_processed_zarr(
            args.zarr, args.state_keys, args.action_keys
        )
    else:
        ep_ends, ee, gripper, cube, action_gripper = load_raw_zarr(args.zarr)

    lengths = episode_lengths(ep_ends)
    n_eps = len(lengths)
    total_steps = int(ep_ends[-1]) if len(ep_ends) > 0 else 0

    # ── episode summary ───────────────────────────────────────────────
    print_section("Episode summary")
    print(f"  Num episodes   : {n_eps}")
    print(f"  Total timesteps: {total_steps}")
    print(f"  Length  min={lengths.min()}  max={lengths.max()}  "
          f"mean={lengths.mean():.1f}  std={lengths.std():.1f}")

    short = np.sum(lengths < 50)
    if short:
        print(f"  [!] {short} episode(s) shorter than 50 steps — consider discarding.")

    # ── idle analysis ─────────────────────────────────────────────────
    print_section("Idle timestep analysis")
    print(f"  (timesteps with movement norm < {args.idle_thresh:.0e} are 'idle')")

    if args.processed:
        for key, arr in {**states, **actions}.items():
            if arr is None:
                continue
            frac = idle_fraction(arr, thresh=args.idle_thresh)
            flag = "  [!] HIGH — consider reviewing" if frac > 0.20 else ""
            print(f"  {key:<30s}  idle fraction = {frac:.2%}{flag}")
    else:
        if ee is not None:
            print(f"  state_ee       idle fraction = {idle_fraction(ee, args.idle_thresh):.2%}")
        if action_gripper is not None:
            # gripper: idle if it barely changes
            diff = np.abs(np.diff(action_gripper.ravel()))
            frac_g = float(np.mean(diff < args.idle_thresh))
            print(f"  action_gripper no-change frac = {frac_g:.2%}")

    # ── per-episode breakdown ─────────────────────────────────────────
    print_section("Per-episode lengths")
    starts = np.concatenate(([0], ep_ends[:-1]))
    for i, (s, e, l) in enumerate(zip(starts, ep_ends, lengths)):
        flag = "  [short!]" if l < 50 else ""
        print(f"  ep {i+1:3d}  steps {s:5d}–{e:5d}  len={l:4d}{flag}")

    # ── state / action statistics ─────────────────────────────────────
    print_section("Array statistics")
    if args.processed:
        for key, arr in {**states, **actions}.items():
            print_array_stats(key, arr)
    else:
        print_array_stats("state_ee", ee)
        print_array_stats("state_gripper", gripper)
        print_array_stats("state_cube", cube)
        print_array_stats("action_gripper", action_gripper)

    # ── plots ─────────────────────────────────────────────────────────
    if args.no_plot:
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[!] matplotlib not available — skipping plots.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Data inspection: {args.zarr.name}", fontsize=13)

    # 1. Episode length histogram
    ax = axes[0, 0]
    ax.hist(lengths, bins=min(n_eps, 30), edgecolor="black")
    ax.axvline(lengths.mean(), color="red", linestyle="--", label=f"mean={lengths.mean():.0f}")
    ax.set_title("Episode length distribution")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Count")
    ax.legend()

    # 2. End-effector XYZ traces (raw) or first state key (processed)
    ax = axes[0, 1]
    if args.processed and "state_ee_xyz" in states and states["state_ee_xyz"] is not None:
        arr = states["state_ee_xyz"]
        for dim, label in enumerate(["x", "y", "z"]):
            ax.plot(arr[:, dim], alpha=0.7, label=label)
        ax.set_title("state_ee_xyz over time")
        ax.legend()
    elif not args.processed and ee is not None:
        for dim, label in enumerate(["x", "y", "z"]):
            ax.plot(ee[:, dim], alpha=0.7, label=label)
        ax.set_title("state_ee (xyz) over time")
        ax.legend()
    else:
        ax.set_visible(False)

    # 3. Action norm per timestep — shows idle stretches
    ax = axes[1, 0]
    if args.processed and "action_ee_xyz" in actions and actions["action_ee_xyz"] is not None:
        arr = actions["action_ee_xyz"]
        norms = np.linalg.norm(arr, axis=-1)
        ax.plot(norms, linewidth=0.5, alpha=0.8)
        ax.axhline(args.idle_thresh, color="red", linestyle="--", label=f"idle thresh={args.idle_thresh:.0e}")
        ax.set_title("action_ee_xyz norm per step (idle = near 0)")
        ax.set_xlabel("Timestep")
        ax.legend()
    elif not args.processed and ee is not None:
        ee_diff = np.linalg.norm(np.diff(ee, axis=0), axis=-1)
        ax.plot(ee_diff, linewidth=0.5, alpha=0.8)
        ax.axhline(args.idle_thresh, color="red", linestyle="--", label=f"idle thresh={args.idle_thresh:.0e}")
        ax.set_title("EE movement per step (idle = near 0)")
        ax.set_xlabel("Timestep")
        ax.legend()
    else:
        ax.set_visible(False)

    # 4. Cube XY positions scatter (data coverage)
    ax = axes[1, 1]
    if args.processed and "state_cube" in states and states["state_cube"] is not None:
        arr = states["state_cube"]
        ax.scatter(arr[:, 0], arr[:, 1], s=1, alpha=0.3)
        ax.set_title("Cube XY position coverage")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
    elif not args.processed and cube is not None:
        ax.scatter(cube[:, 0], cube[:, 1], s=1, alpha=0.3)
        ax.set_title("Cube XY position coverage")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
    else:
        ax.set_visible(False)

    # Draw episode boundaries as vertical lines on the action norm plot
    for ep_end in ep_ends[:-1]:
        axes[1, 0].axvline(ep_end, color="gray", linewidth=0.4, alpha=0.5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
