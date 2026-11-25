"""Run the simulation and analysis steps and generate Figs. 7 and 9.

    python run_all.py                  # everything (about 1 h on a 6-core PC)
    python run_all.py --skip s08       # leave out a step
    python run_all.py --figures-only   # redraw Fig. 7 from stored benchmark results

Each step runs in its own Python process; the options --workers, --force and
--figures-only are passed on to the steps that accept them. Use --render to
show the visualization for each selected step.
"""
import argparse
import os
import subprocess
import sys
import time

from stemarm.paths import RESULTS_DIR

STEPS = [
    ("s01", "s01_ancf_benchmark.py", "4.2  deployment benchmark (Fig. 7)"),
    ("s02", "s02_root_stiffness.py", "4.3  root stiffness K (Fig. 9)"),
    ("s03", "s03_root_damping.py", "4.3  root damping D"),
    ("s04", "s04_deployment_test.py", "5.1  deployment test"),
    ("s05", "s05_ground_slews.py", "4.5/5.1  ground slews, simulations"),
    ("s06", "s06_model_validation.py", "5.2  simulation vs experiment"),
    ("s07", "s07_ground_shaping.py", "5.3  measured shaping results"),
    ("s08", "s08_microgravity_5m.py", "4.4  5 m boom under microgravity"),
    ("s09", "s09_square_path.py", "4.6  square path"),
    ("s10", "s10_slew_extension.py", "4.5  slew with extension"),
    ("s11", "s11_modal_check.py", "5.2  modal check of the ground slews"),
]
ACCEPTS_WORKERS = {"s01", "s04", "s05", "s08", "s09", "s10"}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--skip", nargs="*", default=[], help="steps to leave out, e.g. s08 s09")
    p.add_argument("--only", nargs="*", default=[], help="run only these steps")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--figures-only", action="store_true",
                   help="run only s01 and redraw Fig. 7 from stored benchmark results")
    p.add_argument("--render", action="store_true", help="show each step visualization")
    args = p.parse_args()
    if args.figures_only and ("s01" in args.skip or (args.only and "s01" not in args.only)):
        p.error("--figures-only requires s01")
    here = os.path.dirname(os.path.abspath(__file__))
    start = time.time()
    for key, script, title in STEPS:
        if key in args.skip or (args.only and key not in args.only):
            continue
        if args.figures_only and key != "s01":
            continue
        command = [sys.executable, os.path.join(here, script)]
        if key in ACCEPTS_WORKERS:
            if args.workers:
                command += ["--workers", str(args.workers)]
            if args.force:
                command.append("--force")
            if args.figures_only:
                command.append("--figures-only")
        if args.render:
            command.append("--render")
        print(f"\n=== {key}  {title} ===", flush=True)
        t0 = time.time()
        subprocess.run(command, check=True, cwd=here)
        print(f"=== {key} done in {time.time() - t0:.0f} s ===", flush=True)
    print(f"\nall steps finished in {(time.time() - start) / 60:.1f} min; outputs in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
