"""Small helpers shared by the step scripts."""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed


def add_render_argument(p: argparse.ArgumentParser) -> None:
    p.add_argument("--render", action="store_true", help="show a visualization window")


def parser(description: str, *, figures_only: bool = False):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1),
                   help="number of simulations run in parallel processes")
    p.add_argument("--force", action="store_true", help="rerun simulations that already have results")
    add_render_argument(p)
    if figures_only:
        p.add_argument("--figures-only", action="store_true",
                       help="redraw Fig. 7 from stored benchmark results")
    return p


def run_jobs(function, jobs, workers):
    """Run function(**job) for every job, in parallel processes if workers > 1."""
    if not jobs:
        return
    start = time.time()
    if workers <= 1 or len(jobs) == 1:
        for job in jobs:
            function(**job)
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
            futures = {pool.submit(function, **job): job for job in jobs}
            for future in as_completed(futures):
                future.result()              # re-raise errors of a worker
    print(f"{len(jobs)} simulation(s) finished in {time.time() - start:.0f} s")
    sys.stdout.flush()
