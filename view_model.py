"""Show the manipulator model in the Exudyn renderer while it performs a slew.

    python view_model.py                 # Case B slew, L = 1.0 m, gravity on
    python view_model.py --case A --length 1.2

Nothing is saved. The link and gripper meshes need the numpy-stl package.
The renderer pauses before and after the motion: press SPACE in the render
window to start the motion and again to close the window at the end.
"""
import argparse

from stemarm import params as P, simulate
from stemarm.commands import PointToPoint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", choices=("A", "B"), default="B")
    p.add_argument("--length", type=float, default=1.0, help="deployed length [m]")
    p.add_argument("--no-gravity", action="store_true")
    p.add_argument("--after", type=float, default=5.0, help="time simulated after the command [s]")
    args = p.parse_args()
    spec = P.CASES[args.case]
    command = PointToPoint(spec["q_start_deg"], spec["q_end_deg"], length=args.length)
    simulate.run(command, command.q_start, args.length, command.end_time + args.after,
                 gravity=not args.no_gravity, render=True)


if __name__ == "__main__":
    main()
