"""Augment a multicube zarr dataset into 6 variants per goal colour.

Given N episodes all targeting the same cube colour (e.g. red), produces:

  1. red   (original)
  2. green (relabel: swap pos_cube_red ↔ pos_cube_green, goal [0,1,0])
  3. blue  (relabel: swap pos_cube_red ↔ pos_cube_blue,  goal [0,0,1])
  4. red   + distractor swap (blue ↔ green, goal unchanged)
  5. green + distractor swap
  6. blue  + distractor swap

All 6 variants are written into a single new zarr so that compute_actions.py
can merge them with your other sessions.

Mixed-goal zarrs (e.g. DAgger data collected with --goal-cube all) are
handled automatically: episodes are split by their goal colour, each group
is augmented independently, and the results are concatenated.

Usage:
    python scripts/augment_multicube_dataset.py \\
        --source  datasets/raw/multi_cube/teleop/red-data/so100_multicube_teleop.zarr \\
        --out-dir datasets/raw/multi_cube/teleop/red-data-augmented

    # DAgger data with mixed goals works the same way:
    python scripts/augment_multicube_dataset.py \\
        --source  datasets/raw/multi_cube/dagger/<timestamp>/so100_multicube_teleop.zarr \\
        --out-dir datasets/raw/multi_cube/dagger/<timestamp>-augmented

The script auto-detects whether goals are mixed or uniform.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import zarr


# ── helpers ──────────────────────────────────────────────────────────────────

GOAL_COLOURS = ["red", "green", "blue"]


def _detect_source_colour(state_goal: np.ndarray) -> str:
    """Return the target colour name from a one-hot state_goal array."""
    unique = np.unique(state_goal, axis=0)
    if len(unique) != 1:
        raise ValueError(
            f"Expected all timesteps to share the same goal, got: {unique}"
        )
    hot = unique[0]
    idx = int(np.argmax(hot))
    if hot[idx] != 1.0 or hot.sum() != 1.0:
        raise ValueError(f"state_goal row is not a one-hot vector: {hot}")
    return GOAL_COLOURS[idx]


def _has_mixed_goals(state_goal: np.ndarray) -> bool:
    """Return True if the zarr contains episodes with more than one goal colour."""
    return len(np.unique(state_goal, axis=0)) > 1


def _split_by_colour(
    data: dict[str, np.ndarray],
    ep_ends: np.ndarray,
) -> dict[str, tuple[dict[str, np.ndarray], np.ndarray]]:
    """Split a mixed-goal dataset into per-colour groups.

    Returns a dict mapping colour name → (data_dict, episode_ends) containing
    only the episodes that target that colour.  Episode-end indices are
    re-based to start from 0 within each group.
    """
    starts = np.concatenate(([0], ep_ends[:-1]))
    groups: dict[str, tuple[list[dict[str, np.ndarray]], list[int]]] = {
        c: ([], []) for c in GOAL_COLOURS
    }

    for start, end in zip(starts, ep_ends):
        # Use the first timestep of the episode to determine its goal colour
        ep_goal = data["state_goal"][start]
        idx = int(np.argmax(ep_goal))
        colour = GOAL_COLOURS[idx]

        ep_data = {k: v[start:end] for k, v in data.items()}
        groups[colour][0].append(ep_data)
        groups[colour][1].append(int(end - start))

    result: dict[str, tuple[dict[str, np.ndarray], np.ndarray]] = {}
    for colour, (ep_data_list, ep_lengths) in groups.items():
        if not ep_data_list:
            continue
        merged: dict[str, np.ndarray] = {
            k: np.concatenate([ep[k] for ep in ep_data_list], axis=0)
            for k in ep_data_list[0]
        }
        ends = np.cumsum(ep_lengths, dtype=np.int64)
        result[colour] = (merged, ends)

    return result


def _read_zarr(path: Path) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Return (data_arrays, episode_ends) from a raw multicube zarr."""
    root = zarr.open_group(str(path), mode="r")
    ep_ends = np.array(root["meta"]["episode_ends"])
    n = int(ep_ends[-1])
    data: dict[str, np.ndarray] = {}
    for key in root["data"]:
        data[key] = np.array(root["data"][key][:n])
    return data, ep_ends


def _write_zarr(
    path: Path,
    data: dict[str, np.ndarray],
    episode_ends: np.ndarray,
) -> None:
    """Write augmented data into a fresh zarr at *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    compressor = zarr.codecs.Blosc(cname="zstd", clevel=3, shuffle=2)
    compressors = (compressor,)

    grp_data = root.require_group("data")
    grp_meta = root.require_group("meta")

    for key, arr in data.items():
        grp_data.create_array(key, data=arr.astype(np.float32), compressors=compressors)
    grp_meta.create_array(
        "episode_ends", data=episode_ends.astype(np.int64), compressors=compressors
    )


def _make_variant(
    data: dict[str, np.ndarray],
    *,
    target_colour: str,
    swap_distractors: bool,
) -> dict[str, np.ndarray]:
    """Return a modified copy of *data* for the requested variant.

    Parameters
    ----------
    target_colour : str
        Which cube becomes the goal for this variant ("red", "green", "blue").
    swap_distractors : bool
        If True, also swap the two non-target cube arrays with each other.
    """
    out = {k: v.copy() for k, v in data.items()}

    colours = GOAL_COLOURS.copy()                        # ["red","green","blue"]
    orig_colour = _detect_source_colour(data["state_goal"])
    orig_idx = colours.index(orig_colour)
    tgt_idx  = colours.index(target_colour)

    # ── relabel: swap original-target cube ↔ new-target cube ─────────────
    if orig_colour != target_colour:
        orig_key = f"pos_cube_{orig_colour}"
        tgt_key  = f"pos_cube_{target_colour}"
        out[orig_key], out[tgt_key] = data[tgt_key].copy(), data[orig_key].copy()

    # ── update one-hot goal ───────────────────────────────────────────────
    new_goal = np.zeros(3, dtype=np.float32)
    new_goal[tgt_idx] = 1.0
    out["state_goal"] = np.tile(new_goal, (len(data["state_goal"]), 1))

    # ── optionally swap the two distractor cubes ──────────────────────────
    if swap_distractors:
        distractors = [c for c in colours if c != target_colour]
        k0, k1 = f"pos_cube_{distractors[0]}", f"pos_cube_{distractors[1]}"
        out[k0], out[k1] = out[k1].copy(), out[k0].copy()

    return out


def _concat_variants(
    variants: list[tuple[dict[str, np.ndarray], np.ndarray]],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Concatenate multiple (data, episode_ends) pairs into one dataset."""
    all_data: dict[str, list[np.ndarray]] = {}
    all_ep_ends: list[np.ndarray] = []
    offset = 0

    for data, ep_ends in variants:
        for key, arr in data.items():
            all_data.setdefault(key, []).append(arr)
        all_ep_ends.append(ep_ends + offset)
        offset += int(ep_ends[-1])

    merged_data = {k: np.concatenate(v, axis=0) for k, v in all_data.items()}
    merged_ends = np.concatenate(all_ep_ends, axis=0)
    return merged_data, merged_ends


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="6× augment a single-goal multicube zarr.")
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to source zarr (e.g. datasets/raw/multi_cube/teleop/red-data/so100_multicube_teleop.zarr).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory. The augmented zarr will be written here as so100_multicube_teleop.zarr.",
    )
    parser.add_argument(
        "--no-distractor-swap",
        action="store_true",
        help="Only produce 3 goal-relabeled variants (skip distractor-swap variants).",
    )
    args = parser.parse_args()

    print(f"Reading source zarr: {args.source}")
    data, ep_ends = _read_zarr(args.source)
    n_ep = len(ep_ends)
    n_steps = int(ep_ends[-1])

    swaps = [False, True] if not args.no_distractor_swap else [False]
    all_variants: list[tuple[dict[str, np.ndarray], np.ndarray]] = []
    all_descriptions: list[str] = []

    if _has_mixed_goals(data["state_goal"]):
        # ── mixed-goal zarr (e.g. DAgger with --goal-cube all) ────────────
        colour_groups = _split_by_colour(data, ep_ends)
        colour_ep_counts = {c: len(v[1]) for c, v in colour_groups.items()}
        print(f"  Mixed-goal zarr detected | {n_ep} episodes | {n_steps} steps")
        print(f"  Episodes per colour: " +
              ", ".join(f"{c}={colour_ep_counts.get(c, 0)}" for c in GOAL_COLOURS))

        for src_colour, (grp_data, grp_ends) in colour_groups.items():
            grp_n_ep = len(grp_ends)
            print(f"\n  Augmenting {grp_n_ep} '{src_colour}' episode(s):")
            for target_colour in GOAL_COLOURS:
                for swap in swaps:
                    desc = f"{src_colour}→goal={target_colour}" + (
                        " +distractor_swap" if swap else ""
                    )
                    print(f"    {desc}")
                    v_data = _make_variant(
                        grp_data, target_colour=target_colour, swap_distractors=swap
                    )
                    all_variants.append((v_data, grp_ends.copy()))
                    all_descriptions.append(desc)
    else:
        # ── single-goal zarr (original behaviour) ─────────────────────────
        src_colour = _detect_source_colour(data["state_goal"])
        print(f"  Source colour: {src_colour} | {n_ep} episodes | {n_steps} steps")

        for target_colour in GOAL_COLOURS:
            for swap in swaps:
                desc = f"goal={target_colour}" + (" +distractor_swap" if swap else "")
                print(f"  Generating variant: {desc}")
                v_data = _make_variant(
                    data, target_colour=target_colour, swap_distractors=swap
                )
                all_variants.append((v_data, ep_ends.copy()))
                all_descriptions.append(desc)

    merged_data, merged_ends = _concat_variants(all_variants)
    total_ep = len(merged_ends)
    total_steps = int(merged_ends[-1])

    out_zarr = args.out_dir / "so100_multicube_teleop.zarr"
    print(f"\nWriting {len(all_variants)} variants -> {out_zarr}")
    print(f"  Total episodes: {total_ep}")
    print(f"  Total steps:    {total_steps}")
    _write_zarr(out_zarr, merged_data, merged_ends)
    print("Done.")
    print("\nVariants included:")
    for i, desc in enumerate(all_descriptions):
        print(f"  {i+1}. {desc}")
    print(f"\nNext step: run compute_actions.py pointing --datasets-dir at the parent of this zarr's folder,")
    print(f"or add {args.out_dir} as an additional source directory.")


if __name__ == "__main__":
    main()
