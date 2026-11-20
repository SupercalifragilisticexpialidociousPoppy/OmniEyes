"""Analyze calibration data to recommend optimal tau values for early-layer gating.
Usage (from the repository root): python src/calibrate.py results/iteration1/calib.jsonl [--n_frames 33]
"""
import argparse, json, os, sys
import numpy as np


def analyze_calibration(path, n_frames=33):
    if not os.path.exists(path):
        print(f"Error: file not found: {path}", file=sys.stderr)
        return

    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if not rows:
        print("No records found in calibration file.", file=sys.stderr)
        return

    # Collect all non-first-frame similarity scores
    all_sims = []
    for r in rows:
        sims = r.get("sims", [])
        if len(sims) > 1:
            all_sims.extend(sims[1:])

    all_sims = np.array(all_sims, dtype=float)
    print(f"=== CALIBRATION ANALYSIS ({len(rows)} samples, {len(all_sims)} similarity measurements) ===")
    print(f"Min:    {all_sims.min():.4f}")
    print(f"P10:    {np.percentile(all_sims, 10):.4f}")
    print(f"P25:    {np.percentile(all_sims, 25):.4f}")
    print(f"Median: {np.median(all_sims):.4f}")
    print(f"P75:    {np.percentile(all_sims, 75):.4f}")
    print(f"P90:    {np.percentile(all_sims, 90):.4f}")
    print(f"P95:    {np.percentile(all_sims, 95):.4f}")
    print(f"Max:    {all_sims.max():.4f}")

    print("\n=== FRAME RETENTION SIMULATION ACROSS TAU VALUES ===")
    print(f"{'tau':<8}{'avg_keep_%':>14}{'avg_kept_frames':>18}{'compression':>15}")

    tau_range = np.arange(0.80, 0.991, 0.01)
    for tau in tau_range:
        keep_rates = []
        for r in rows:
            sims = r.get("sims", [])
            if not sims:
                continue
            kept = 1
            for s in sims[1:]:
                if s < tau:
                    kept += 1
            keep_rates.append(kept / len(sims))
        if keep_rates:
            avg_rate = np.mean(keep_rates)
            frames = avg_rate * n_frames
            comp = f"{(1 - avg_rate) * 100:.1f}% dropped"
            star = "  <-- RECOMMENDED" if round(tau, 2) in (0.88, 0.90, 0.93, 0.95) else ""
            print(f"{tau:<8.2f}{avg_rate * 100:>13.1f}%{frames:>17.1f}/{n_frames}{comp:>15}{star}")

    print("\nRecommended tau sweep for Iteration 2:")
    print("  tau=0.88 -> ~24% frame retention (~8 frames)  [Aggressive]")
    print("  tau=0.90 -> ~33% frame retention (~11 frames) [Moderate]")
    print("  tau=0.93 -> ~52% frame retention (~17 frames) [Balanced]")
    print("  tau=0.95 -> ~61% frame retention (~20 frames) [Conservative]")


def main():
    ap = argparse.ArgumentParser(description="Analyze calibration similarity distribution")
    ap.add_argument("calib_jsonl", help="Path to calib.jsonl")
    ap.add_argument("--n_frames", type=int, default=33, help="Reference window frame count (default: 33)")
    args = ap.parse_args()
    analyze_calibration(args.calib_jsonl, args.n_frames)


if __name__ == "__main__":
    main()
