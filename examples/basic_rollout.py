"""Minimal 24-hour blood glucose rollout with GlucoSim.

Runs one seeded day for a T1D virtual patient with a no-intervention
policy (the patient still receives basal insulin and eats scheduled
scenario meals), then prints standard glycemic metrics.

Run with:
    python examples/basic_rollout.py

Expected behavior (sanity checks):
  * CGM starts near a fasting value (roughly 100-160 mg/dL).
  * CGM rises after scenario meals and stays within ~40-400 mg/dL.
  * The same --seed always reproduces the same trajectory.
"""
import argparse
import os

# Keep the example light-weight and deterministic across machines.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

import glucosim  # noqa: F401  (registers t1d-v0 / t2d-v0 / t2d_no_pump-v0)
from glucosim import gym_env as gym


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="t1d-v0", choices=["t1d-v0", "t2d-v0", "t2d_no_pump-v0"])
    parser.add_argument("--patient", default="adolescent#001")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--plot", default=None, help="Optional path to save a CGM plot (PNG)")
    args = parser.parse_args()

    env = gym.make(
        args.env_id,
        simulation_minutes=args.hours * 60,
        sample_time=5,
        patient_name=args.patient,
    )

    print(f"Resetting {args.env_id} for patient {args.patient} (seed={args.seed})...")
    print("(the first reset JIT-compiles the simulator and may take a minute)")
    obs, info = env.reset(seed=args.seed)

    cgm_trace = [obs[0]]
    total_reward = 0.0
    total_cost = 0.0
    noop = np.array([0, 0])  # [bolus_level, meal_level]; 0 = no intervention

    while True:
        obs, reward, cost, terminated, truncated, info = env.step(noop)
        cgm_trace.append(obs[0])
        total_reward += reward
        total_cost += cost
        if terminated or truncated:
            break

    cgm = np.asarray(cgm_trace)
    tir = float(np.mean((cgm >= 70.0) & (cgm <= 180.0)))
    print(f"\nEpisode finished after {len(cgm) - 1} steps "
          f"({'terminated' if terminated else 'truncated at horizon'})")
    print(f"  CGM mean / min / max : {cgm.mean():6.1f} / {cgm.min():6.1f} / {cgm.max():6.1f} mg/dL")
    print(f"  Time in range 70-180 : {100 * tir:5.1f} %")
    print(f"  Time hypo (<70)      : {100 * float(np.mean(cgm < 70.0)):5.1f} %")
    print(f"  Time hyper (>180)    : {100 * float(np.mean(cgm > 180.0)):5.1f} %")
    print(f"  Episode return       : {total_reward:8.2f}")
    print(f"  Episode safety cost  : {total_cost:8.2f}")

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        hours = np.arange(len(cgm)) * 5 / 60.0
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(hours, cgm, label="CGM")
        ax.axhline(70, color="red", linestyle="--", label="hypo threshold (70)")
        ax.axhline(180, color="orange", linestyle="--", label="hyper threshold (180)")
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("CGM (mg/dL)")
        ax.set_title(f"{args.env_id} / {args.patient} / seed {args.seed}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"  Saved plot to {args.plot}")

    env.close()


if __name__ == "__main__":
    main()
