"""Step 5 - Simulations of the ground slews (Sections 5.2 and 5.3).

Case A: q1 = 0 deg, q2 from 10 to 90 deg.  Case B: q1 from 0 to 90 deg and q2
from 10 to 90 deg. L = 1.0 m, gravity on. Each case is run unshaped and with
ZVD (3.5 Hz) and ZVD2 (3.5 + 4.0 Hz) shapers, and continued for 20 s after the
command ends. The unshaped runs are used by s06, s10 and s11; the shaped runs
are not used in the paper.

Results: results/s05_ground_slews/case_<A|B>_<shaper>/
Run time: about 20 s per case (six cases in parallel).
"""
import os

from stemarm import params as P, paths, shaping, simulate
from stemarm.commands import PointToPoint
from stemarm.runner import parser, run_jobs

SHAPER_FREQUENCIES = (3.5, 4.0)    # Hz, measured dominant frequency and the next 0.5 Hz line
SHAPER_ZETA = 0.01
POST_COMMAND_TIME = 20.0           # s


def case_folder(case, shaper_name):
    return paths.result_dir("s05_ground_slews", f"case_{case}_{shaper_name}")


def simulate_case(case, shaper_name, joint_drive="state", force=False, render: bool = False):
    folder = case_folder(case, shaper_name)
    if simulate.is_done(folder) and not (force or render):
        print(f"  case {case} {shaper_name}: stored result used")
        return
    shaper = shaping.make_shaper(shaper_name, SHAPER_FREQUENCIES, SHAPER_ZETA)
    spec = P.CASES[case]
    command = PointToPoint(spec["q_start_deg"], spec["q_end_deg"], shaper=shaper, length=1.0)
    print(f"  case {case} {shaper_name}: command ends at {command.end_time:.4f} s")
    result = simulate.run(command, command.q_start, 1.0, command.end_time + POST_COMMAND_TIME,
                          gravity=True, joint_drive=joint_drive, verbose=0,
                          render=render, render_pause=False)
    simulate.save(result, folder, {
        "case": case, "shaper": shaper_name, "frequencies_hz": list(SHAPER_FREQUENCIES),
        "zeta": SHAPER_ZETA, "command_end_s": command.end_time, "length_m": 1.0,
        "gravity": True, "joint_drive": joint_drive})


def main():
    p = parser(__doc__)
    p.add_argument("--joint-drive", choices=("state", "constraint"), default="state",
                   help="how q(t) is imposed; 'state' reproduces the paper (see USER_MANUAL.md)")
    args = p.parse_args()
    jobs = [dict(case=c, shaper_name=s, joint_drive=args.joint_drive, force=args.force, render=args.render)
            for c in ("A", "B") for s in ("none", "zvd", "zvd2")]
    run_jobs(simulate_case, jobs, 1 if args.render else args.workers)


if __name__ == "__main__":
    main()
