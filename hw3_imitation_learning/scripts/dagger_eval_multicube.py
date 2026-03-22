"""DAgger interactive evaluation for the multicube goal-conditioned scene.

Runs policy inference in the multicube scene.  At any time you (the expert)
can press the takeover key to assume manual control.  While in takeover mode
every timestep is recorded into a zarr store in the same format as
record_teleop_demos.py --multicube, so it can be merged and retrained on via
compute_actions.py.

Goal cycles through red → green → blue across episodes by default.
Use --goal-cube to fix a single colour.

Collected data is saved under datasets/raw/multi_cube/dagger/<timestamp>/ and
can be merged with your original demonstrations for retraining:

    python scripts/compute_actions.py \\
        --datasets-dir datasets/raw/multi_cube \\
        --out datasets/processed/multi_cube/processed_ee_xyz.zarr \\
        --action-space ee_xyz

Usage:
    python scripts/dagger_eval_multicube.py \\
        --checkpoint checkpoints/multi_cube/best_model_ee_xyz_multitask.pt \\
        --num-episodes 15
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import torch
from hw3.dataset import Normalizer
from hw3.eval_utils import (
    apply_action,
    check_cube_out_of_bounds,
    check_success,
    check_wrong_cube_in_bin,
    infer_action_chunk,
    load_checkpoint,
)
from hw3.sim_env import CUBE_COLORS, SO100MulticubeSimEnv
from hw3.teleop_utils import (
    CAMERA_NAMES,
    DEFAULT_KEYMAP_PATH,
    compose_camera_views,
    handle_teleop_key,
    load_keymap,
)
import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from record_teleop_demos import MulticubeZarrWriter
from so101_gym.constants import ASSETS_DIR

XML_PATH_MULTICUBE = ASSETS_DIR / "so100_multicube_ee.xml"

CUBE_FREE_DIM = 7
ALL_CUBES_DIM = 3 * CUBE_FREE_DIM  # 21


# ── recording helper ─────────────────────────────────────────────────


def _record_multicube_step(
    env: SO100MulticubeSimEnv,
    writer: MulticubeZarrWriter,
) -> None:
    """Append one timestep of multicube state to the writer."""
    joints = env.get_joint_angles()
    ee_state = env.get_ee_state()
    all_cubes = env.get_all_cubes_state()          # (21,) red+green+blue
    gripper_state = np.array([env.get_gripper_angle()], dtype=np.float32)
    action_gripper = np.array(
        [env.data.ctrl[env.act_ids[env._jaw_idx]]], dtype=np.float32
    )
    obstacle_state = np.zeros(3, dtype=np.float32)
    state_goal = env.get_goal_onehot()
    goal_pos = env.get_goal_pos()

    writer.append_with_goal(
        joints,
        ee_state,
        all_cubes,
        gripper_state,
        action_gripper,
        obstacle_state,
        state_goal,
        goal_pos,
    )


# ── main DAgger loop ─────────────────────────────────────────────────


def run_dagger_episode(
    env: SO100MulticubeSimEnv,
    model: torch.nn.Module,
    normalizer: Normalizer,
    state_keys: list[str],
    action_keys: list[str],
    device: torch.device,
    writer: MulticubeZarrWriter,
    key_to_action: dict[int, str],
    *,
    max_steps: int = 800,
    successes: int = 0,
    total: int = 0,
) -> tuple[bool, int, bool, bool]:
    """Run one multicube DAgger episode.

    Returns (success, n_takeover_steps, aborted, replay).
    """
    rng_state_before_reset = env.rng.bit_generator.state
    obs = env.reset()

    action_queue: list[np.ndarray] = []
    step = 0
    success = False
    human_control = False
    n_takeover_steps = 0
    recording_this_episode = False

    GRACE_SECS = 1.7
    grace_steps_remaining: int | None = None

    while step < max_steps or human_control:
        k_raw = cv2.waitKeyEx(1)

        if k_raw != -1:
            action_name = key_to_action.get(k_raw)

            if action_name == "escape":
                if recording_this_episode:
                    writer.discard_episode()
                    print("  Episode discarded on escape.")
                return success, n_takeover_steps, True, False

            if action_name == "record":
                human_control = not human_control
                if human_control:
                    print(
                        f"  >>> HUMAN TAKEOVER (goal: {env.goal_cube}) — "
                        "press 'record' again to hand back to policy"
                    )
                    action_queue.clear()
                    recording_this_episode = True
                else:
                    print("  <<< POLICY RESUMED")

            if action_name == "reset":
                if recording_this_episode:
                    writer.discard_episode()
                    print("  Episode discarded — replaying same scenario.")
                env.rng.bit_generator.state = rng_state_before_reset
                return False, 0, False, True  # replay=True

            if k_raw in (13, 0x0D):
                if recording_this_episode:
                    writer.discard_episode()
                    print("  Episode discarded — skipping to next.")
                return False, 0, False, False

            if human_control and action_name is not None:
                handle_teleop_key(
                    action_name,
                    env.data,
                    env.model,
                    env.mocap_id,
                    env.act_ids[env._jaw_idx],
                )

        # ── record state BEFORE step (human control only) ────────────
        if human_control:
            _record_multicube_step(env, writer)
            n_takeover_steps += 1

        # ── policy inference ──────────────────────────────────────────
        if not human_control:
            if not action_queue:
                chunk = infer_action_chunk(
                    model=model,
                    normalizer=normalizer,
                    obs=obs,
                    state_keys=state_keys,
                    device=device,
                )
                action_queue.extend(chunk)
            action = action_queue.pop(0)
            apply_action(env, action, action_keys)

        obs = env.step()
        step += 1

        # ── termination checks ────────────────────────────────────────
        success = check_success(env)
        wrong_in_bin = check_wrong_cube_in_bin(env)

        if success:
            if human_control and grace_steps_remaining is None:
                grace_steps_remaining = int(GRACE_SECS / env.dt_ctrl)
                print(
                    f"  [{env.goal_cube}] Cube in bin!  Recording "
                    f"{grace_steps_remaining} more steps ({GRACE_SECS}s grace)..."
                )
            elif not human_control:
                if recording_this_episode:
                    writer.end_episode()
                    print(
                        f"  DAgger episode saved [{env.goal_cube}] "
                        f"({n_takeover_steps} takeover steps)"
                    )
                return True, n_takeover_steps, False, False

        if grace_steps_remaining is not None:
            grace_steps_remaining -= 1
            if grace_steps_remaining <= 0:
                if recording_this_episode:
                    writer.end_episode()
                    print(
                        f"  DAgger episode saved [{env.goal_cube}] "
                        f"({n_takeover_steps} takeover steps)"
                    )
                return True, n_takeover_steps, False, False

        if check_cube_out_of_bounds(env):
            print(f"  [{env.goal_cube}] Cube out of bounds — early termination.")
            if recording_this_episode:
                writer.end_episode()
                print(f"  DAgger episode saved ({n_takeover_steps} takeover steps)")
            return False, n_takeover_steps, False, False

        # ── render ────────────────────────────────────────────────────
        img = compose_camera_views({cam: env.render(cam) for cam in CAMERA_NAMES})

        status = f"[{env.goal_cube.upper()}] Step {step}/{max_steps}"
        status += " | HUMAN" if human_control else f" | POLICY (queue {len(action_queue)})"
        cv2.putText(img, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        sr_text = (
            f"Success: {successes}/{total} ({successes/total*100:.0f}%)"
            if total > 0 else "Success: -/-"
        )
        cv2.putText(img, sr_text, (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 0) if success else (0, 0, 255), 2)

        dagger_text = (
            f"DAgger steps: {n_takeover_steps} | Episodes saved: {writer.num_episodes}"
        )
        cv2.putText(img, dagger_text, (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 200, 0), 2)

        if wrong_in_bin:
            cv2.putText(img, f"WRONG CUBE IN BIN: {wrong_in_bin}!", (10, 165),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        mode_text = "HUMAN" if human_control else "POLICY"
        mode_color = (0, 0, 255) if human_control else (0, 255, 0)
        cv2.putText(img, mode_text, (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, mode_color, 3)

        def _label_for(act: str) -> str:
            for code, a in key_to_action.items():
                if a == act:
                    if 32 <= (code & 0xFF) <= 126:
                        ch = chr(code & 0xFF)
                        return ch if ch.strip() else "SPACE"
                    if code & 0xFF == 27:
                        return "ESC"
                    return f"key:{code}"
            return "?"

        hint = (
            f"{_label_for('record')} takeover | "
            f"{_label_for('reset')} replay | "
            f"ENTER skip | "
            f"{_label_for('escape')} quit"
        )
        cv2.putText(img, hint, (10, img.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("DAgger Multicube", img)
        time.sleep(env.dt_ctrl)

    # Reached max_steps
    if recording_this_episode:
        writer.end_episode()
        print(f"  DAgger episode saved ({n_takeover_steps} takeover steps)")
    return success, n_takeover_steps, False, False


# ── entry point ───────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DAgger interactive evaluation for the multicube scene."
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Path to the multitask model checkpoint (.pt).",
    )
    parser.add_argument(
        "--num-episodes", type=int, default=15,
        help="Number of evaluation episodes (default: 15).",
    )
    parser.add_argument(
        "--max-steps", type=int, default=800,
        help="Maximum steps per episode (default: 800).",
    )
    parser.add_argument(
        "--goal-cube", type=str, default="all",
        choices=["red", "green", "blue", "all"],
        help="Goal colour ('all' cycles red→green→blue evenly, default: all).",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducible cube spawns.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory for DAgger zarr (default: datasets/raw/multi_cube/dagger/<timestamp>).",
    )
    parser.add_argument(
        "--keymap", type=Path, default=None,
        help="Path to keymap.json (default: hw3/keymap.json).",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, normalizer, chunk_size, state_keys, action_keys = load_checkpoint(
        args.checkpoint, device
    )

    use_mocap = not any("action_joints" in k for k in action_keys)

    # Build goal schedule
    if args.goal_cube == "all":
        goal_schedule = [CUBE_COLORS[i % len(CUBE_COLORS)] for i in range(args.num_episodes)]
    else:
        goal_schedule = [args.goal_cube] * args.num_episodes

    env = SO100MulticubeSimEnv(
        xml_path=XML_PATH_MULTICUBE,
        render_w=640,
        render_h=480,
        use_mocap=use_mocap,
        goal_cube=goal_schedule[0],
        shuffle_cubes=True,
        seed=args.seed,
    )

    km_path = args.keymap or DEFAULT_KEYMAP_PATH
    key_to_action = load_keymap(km_path)
    print(f"Loaded keymap from {km_path}")

    # DAgger output zarr
    if args.output_dir:
        out_dir = args.output_dir
    else:
        ts = datetime.now(ZoneInfo("Europe/Berlin")).strftime("%Y-%m-%d_%H-%M-%S")
        out_dir = Path("./datasets/raw/multi_cube/dagger") / ts
    out_zarr = out_dir / "so100_multicube_teleop.zarr"
    print(f"DAgger data will be saved to: {out_zarr}")

    writer = MulticubeZarrWriter(
        path=out_zarr,
        joint_dim=6,
        ee_dim=7,
        cube_dim=0,   # multicube writer stores cubes in per-colour arrays
        gripper_dim=1,
        obstacle_dim=3,
        flush_every=12,
    )
    writer.set_attrs(
        num_dagger_episodes=0,  # updated at end
        goal_cube=args.goal_cube,
    )

    cv2.namedWindow("DAgger Multicube", cv2.WINDOW_AUTOSIZE)

    successes = 0
    total_takeover_steps = 0
    try:
        ep = 0
        while ep < args.num_episodes:
            goal = goal_schedule[ep]
            env.set_goal(goal)
            ep += 1
            print(f"\n═══ DAgger Episode {ep}/{args.num_episodes}  (goal: {goal}) ═══")
            print("  Policy is running. Press your 'record' key to take over control.")

            success, n_takeover, aborted, replay = run_dagger_episode(
                env, model, normalizer, state_keys, action_keys, device,
                writer, key_to_action,
                max_steps=args.max_steps,
                successes=successes,
                total=ep - 1,
            )

            if aborted:
                print("Aborted by user.")
                break

            if replay:
                print("  Replaying same episode...")
                ep -= 1
                continue

            total_takeover_steps += n_takeover
            if success:
                successes += 1
            rate = successes / ep * 100
            result = "SUCCESS" if success else "FAIL"
            print(f"Episode {ep}: {result} | takeover steps: {n_takeover}")
            print(f"  Success rate so far: {successes}/{ep} ({rate:.0f}%)")

    finally:
        writer.set_attrs(num_dagger_episodes=writer.num_episodes)
        writer.flush()
        cv2.destroyAllWindows()

    n_eps = writer.num_episodes
    n_steps = writer.num_steps_total
    rate = successes / max(1, args.num_episodes) * 100
    print(f"\n{'=' * 55}")
    print("DAgger multicube session complete.")
    print(f"  Episodes evaluated:    {args.num_episodes}")
    print(f"  Success rate:          {successes}/{args.num_episodes} ({rate:.0f}%)")
    print(f"  Total takeover steps:  {total_takeover_steps}")
    print(f"  DAgger episodes saved: {n_eps} ({n_steps} total steps)")
    print(f"  Data saved to:         {out_zarr}")
    if n_eps > 0:
        print("\nNext steps:")
        print("  1. Run compute_actions.py to merge DAgger data with your existing dataset")
        print("  2. Retrain with the merged zarr using train.py")


if __name__ == "__main__":
    main()
