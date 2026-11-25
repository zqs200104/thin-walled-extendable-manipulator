# Variable-length space manipulator with a thin-walled deployable boom: simulation code and data

Code and measured data for the paper

> Quansheng Zhang et al., *Flexible dynamics modeling for model-based vibration control of a variable-length space manipulator.*

Citation details will be added after publication.

The manipulator has two revolute joints (R1, R2) and a prismatic joint (T1) that
deploys a thin-walled tubular boom. The flexible multibody model is built in
[Exudyn](https://github.com/jgerstmayr/EXUDYN):

* the joint chain is a kinematic tree read from the URDF in `model/`;
* the boom is discretized with three-dimensional ANCF beam elements whose
  reference length follows the deployed length;
* the boom root is a generic joint that leaves only one rotation free, with a
  linear torsional spring-damper (K, D) identified on the prototype;
* the end-effector is a rigid body fixed to the tip node.

The model predicts the dominant vibration frequency, from which ZVD and
two-frequency ZVD<sub>2</sub> input shapers are designed for point-to-point slews and,
through path-progress shaping, for trajectory motion.

## Quick start

```bash
conda env create -f environment.yml      # or: pip install -r requirements.txt  (Python 3.13)
conda activate stemarm
python run_all.py                        # all steps, about 1 h on a 6-core PC
```

The scripts generate Figs. 7 and 9 in `results/figures/` (PDF and PNG).
Simulation files, tables, and metrics go to `results/<step>/`.
See **[USER_MANUAL.md](USER_MANUAL.md)**
for the installation, every step, the parameters and the file formats, worked
examples of the library (Section 7) and a reference of its functions (Section 8).
The manual also includes [two simulation animations](USER_MANUAL.md#78-simulation-animations).
An [offline PDF manual](USER_MANUAL.pdf) is available.

## Steps

Each script runs the computations for a part of the paper. The manual lists
which steps use results from an earlier simulation.

| Script | Paper | Output | Run time |
| --- | --- | --- | --- |
| `s01_ancf_benchmark.py` | 4.2 | Fig. 7; RMS deviations 0.67 / 0.26 mm | 2 min |
| `s02_root_stiffness.py` | 4.3 | Fig. 9; K = 64.83 N m/rad | 1 min |
| `s03_root_damping.py` | 4.3 | D = 0.1541 N m s/rad from the 420 g free decay | 1 min |
| `s04_deployment_test.py` | 5.1 | simulated responses and frequency versus deployed length | 3 min |
| `s05_ground_slews.py` | 5.2, 5.3 | simulations of Cases A and B (used by s06, s10, s11) | 1-2 min |
| `s06_model_validation.py` | 5.2 | Table 4 and simulation-to-measurement metrics | < 1 min |
| `s07_ground_shaping.py` | 5.3 | measured J<sub>RMS</sub> reductions and joint excitation | < 1 min |
| `s08_microgravity_5m.py` | 4.4 | simulated slews and residual metrics for a 5 m boom | 35 min |
| `s09_square_path.py` | 4.6 | Table 3, frozen-pose modes, and corner-rest comparisons | 8 min |
| `s10_slew_extension.py` | 4.5 | simulated slew with extension and residual metrics | 1 min |
| `s11_modal_check.py` | 5.2 (check) | complex modes and fitted frequency and damping of the slews | 1 min |

Add `--render` to any step to open its visualization. For example, run
`python s05_ground_slews.py --render` for the ground slews or
`python s07_ground_shaping.py --render` for the measured residuals. The
[user manual](USER_MANUAL.md#visualization-with---render) lists what each
step displays and how to install the optional mesh dependency.

`view_model.py` shows the model in the Exudyn renderer during a slew. The STL
meshes in `model/` are simple placeholder shapes used only for this display;
the masses, inertias and joint frames of the model come from the URDF and
`stemarm/params.py`, so the meshes do not affect any result.

## Layout

```
stemarm/      model and analysis library (parameters, model, shapers, commands, evaluation)
model/        URDF and placeholder display meshes of the manipulator
data/         measured data (see data/README.md)
s01-s11       one script per subsection of the paper
run_all.py    runs all steps in order
docs/media/   animations embedded in the manual
results/      created by the scripts (not tracked)
```

## License

To be added by the authors.
