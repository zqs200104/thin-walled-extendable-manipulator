# Measured data

All measurements were taken on the prototype described in Section 4.1 of the paper.
Positions are in millimetres in the frame of the motion-capture system (NOKOV,
eight cameras, 90 Hz), which differs from the simulation frame only by the
offset `MOCAP_ORIGIN_OFFSET_MM` in `stemarm/params.py`.

| Folder | Content | Used by |
| --- | --- | --- |
| `park2013/` | Tip response digitized from Fig. 3 of Park et al. (2013) | `s01_ancf_benchmark.py` |
| `root_identification/static_tip_deflection.csv` | 24 static loading tests (3 tip masses x 8 deployed lengths) | `s02_root_stiffness.py` |
| `root_identification/free_decay_420g/` | Three free-decay tests, L = 1 m, 420 g tip mass | `s03_root_damping.py` |
| `deployment_test/` | Extension to 1 m, hold and retraction at q1 = 0 deg, q2 = 90 deg (two runs, two markers each) | `s04_deployment_test.py` (run 2, `SinglePoint1.ly`) |
| `ground_tests/` | Earlier ground slews of Cases A and B (2026-08-27 and 2026-09-08) | `s11_modal_check.py` |
| `shaping_tests/` | Ground slews of Cases A and B (2026-11-13), unshaped and ZVD 3.1 Hz | `s06_model_validation.py` (unshaped, first trial), `s07_ground_shaping.py` |

## Ground slews

Each trial folder holds two files:

* `SinglePoint0.ly`: motion-capture track of marker point P on the boom tip;
* `robot_feedback_<date>_<time>.csv`: log of the onboard controller.

Case A: q1 = 0 deg, q2 from 10 to 90 deg. Case B: q1 from 0 to 90 deg and q2
from 10 to 90 deg (both joints start and stop together).

**`ground_tests/case_<A|B>/<none|zvd|zvd2>/<trial>/`**. The
unshaped slews were recorded on 2026-08-27; `s11` uses the first trial of each
case. Case B, unshaped, trial 3 contains no motion. The `zvd` (3.5 Hz) and `zvd2`
(3.5 + 4.0 Hz) folders hold shaped slews recorded on 2026-09-08; they are kept as
raw data and are not used in the paper (in Case A, ZVD, trial 2 the
motion-capture and controller clocks are offset by 0.6045 s).

**`shaping_tests/case_<A|B>/<none|zvd_3.1>/<trial>/`** (Sections 5.2 and 5.3),
recorded on 2026-11-13. `s06` compares the first unshaped trial of each case
with the model (Table 4). `s07` reports residual metrics for all trials.
ZVD shaper designed for 3.1 Hz, the frequency predicted by the model, with a
damping ratio of 0.01; three trials per command. Recording times (controller
log): Case B unshaped 1 at 09:23, ZVD 1 at 09:27, unshaped 2-3 at 12:16-12:20,
ZVD 2-3 at 12:25-12:30; Case A unshaped 1 at 10:02, ZVD 1 at 10:07, unshaped 2-3
at 10:49-10:51, ZVD 2-3 at 10:55-10:58. Other commands of that day (ZVD 3.5 Hz,
ZVD 3.6 Hz, ZVD2 3.1 + 3.6 Hz, one trial each) are not included. Every record
has a resting stretch before the slew and a controller log.

The joint motors were run in position-velocity mode with the gains
q1: velocity kp = 0.008, ki = 0.0005, position kp = 1000;
q2: velocity kp = 0.0038, ki = 0.0005, position kp = 450.

## File formats

**`.ly`** (NOKOV export, tab separated). Line 2 gives the data rate and the
number of frames; from line 6 on, each row holds frame number, time [s],
timestamp [ms], X, Y, Z [mm], and the velocity and acceleration estimated by
the exporter (empty where it could not compute them).

**`robot_feedback_*.csv`** (comma separated, UTF-8 with BOM). Main columns:
`t_s` time; `q1_deg`, `q2_deg` encoder angles; `dq1_deg_s`, `dq2_deg_s`
rates; `L_m` deployed length; `tau1_Nm`, `tau2_Nm` joint torques;
`q1_ref_deg`, `q2_ref_deg`, `dq1_ref_deg_s`, `dq2_ref_deg_s` commanded values;
`e_q1_deg`, `e_q2_deg` tracking errors.

**`static_tip_deflection.csv`**: tip mass [g], deployed length [cm] and the
tip displacement along gravity caused by the mass [mm].

**`park2013/fig3_tip_displacement_digitized.csv`**: time [s] and transverse
tip displacement [mm] of the reference solution.

Reference: S. Park, H.H. Yoo, J. Chung, Vibrations of an axially moving beam
with deployment or retraction, AIAA Journal 51(3) (2013) 686-696,
doi:10.2514/1.J052059.
