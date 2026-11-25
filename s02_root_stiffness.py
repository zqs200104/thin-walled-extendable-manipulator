"""Step 2 - Identification of the root stiffness K (Section 4.3, Fig. 9).

Static tests: the boom is horizontal (mounting bias beta), a mass m is added
at the tip of the deployed length L, and the tip displacement along gravity
is measured (data/root_identification/static_tip_deflection.csv, 24 tests).

1. Fit, Eqs. (51)-(53): with the beam and axial compliances known,
       delta_root = delta - F cos^2(beta) L^3/(3 EI) - F sin^2(beta) L/(EA),
       lambda     = F cos^2(beta) L^2,
   K = 1/alpha from the least-squares line delta_root = alpha * lambda through the origin.
   The fit is repeated for each tip mass as a check.
2. Validation: each test is simulated statically with the ANCF boom (48
   elements) on the root spring K, and the simulated deflection is compared
   with the measurement.

Outputs: results/s02_root_stiffness/ and Fig. 9 (static_root_stiffness_validation).
Run time: under 1 min.
"""
import argparse
import csv
import os

import numpy as np
import exudyn as exu
from exudyn.itemInterface import MarkerNodeCoordinate, LoadCoordinate, SensorNode

from stemarm import params as P, paths, plotstyle, rig
from stemarm.rendering import live
from stemarm.runner import add_render_argument
from stemarm.section import section_properties

N_ELEMENTS = P.N_ELEMENTS   # 48, as in all simulations (Table 2)
G = rig.G
OUT = paths.result_dir("s02_root_stiffness")
DATA = os.path.join(paths.DATA_DIR, "root_identification", "static_tip_deflection.csv")
PROPS = section_properties()
BIAS = np.deg2rad(P.BIAS_DEG)


def load_tests():
    rows = np.loadtxt(DATA, delimiter=",", comments="#", skiprows=4)
    return [{"mass_g": m, "L_cm": l, "delta_mm": d} for m, l, d in rows]


def fit_stiffness(tests):
    """Least-squares fit of K through the origin, Eqs. (51)-(53)."""
    F = np.array([t["mass_g"] / 1000.0 * G for t in tests])
    L = np.array([t["L_cm"] / 100.0 for t in tests])
    delta = np.array([t["delta_mm"] / 1000.0 for t in tests])
    cos2, sin2 = np.cos(BIAS)**2, np.sin(BIAS)**2
    beam = F * (cos2 * L**3 / (3.0 * PROPS["EI_z"]) + sin2 * L / PROPS["EA"])
    x = cos2 * F * L**2
    alpha = np.sum(x * (delta - beam)) / np.sum(x**2)
    return 1.0 / alpha


def simulate_static(length, mass_kg, k_root, *, render: bool = False):
    """Tip displacement along gravity [mm] of the loaded boom (no self-weight: tests are differences)."""
    sc = exu.SystemContainer()
    mbs = sc.AddSystem()
    boom = rig.build(mbs, length, N_ELEMENTS, k_root, props=PROPS, with_graphics=render)
    force = mass_kg * np.array([0.0, -G, 0.0])
    for c in range(3):
        marker = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=boom["tip"], coordinate=c))
        mbs.AddLoad(LoadCoordinate(markerNumber=marker, load=float(force[c])))
    sensor = mbs.AddSensor(SensorNode(nodeNumber=boom["tip"], storeInternal=True,
                                      outputVariableType=exu.OutputVariableType.Position))
    mbs.Assemble()
    with live(sc, render, title=f"s02: L = {length:g} m, tip mass = {mass_kg * 1000:g} g"):
        mbs.SolveStatic(rig.static_settings(), updateInitialValues=True)
    tip = np.array(mbs.GetSensorValues(sensor), dtype=float)
    return -1000.0 * float((tip - (boom["p_root"] + length * boom["ex"]))[1])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_render_argument(p)
    args = p.parse_args()
    plotstyle.apply()
    tests = load_tests()
    k_root = fit_stiffness(tests)
    print(f"  E I_zeta = {PROPS['EI_z']:.2f} N m^2, beta = {P.BIAS_DEG} deg")
    print(f"  identified K = {k_root:.4f} N m/rad (value used in all simulations: {P.K_ROOT:.4f})")
    for mass in (220.0, 320.0, 420.0):
        print(f"    fit with {mass:.0f} g only: K = {fit_stiffness([t for t in tests if t['mass_g'] == mass]):.2f}")

    preview = max(tests, key=lambda t: (t["L_cm"], t["mass_g"])) if args.render else None
    rows = []
    for t in tests:
        sim = simulate_static(t["L_cm"] / 100.0, t["mass_g"] / 1000.0, k_root, render=t is preview)
        rows.append([t["mass_g"], t["L_cm"], t["delta_mm"], sim, sim - t["delta_mm"],
                     100.0 * (sim - t["delta_mm"]) / t["delta_mm"]])
    rows = np.array(rows)
    err, exp = rows[:, 4], rows[:, 2]
    r2 = 1.0 - np.sum(err**2) / np.sum((exp - exp.mean())**2)
    rmse, mae = np.sqrt(np.mean(err**2)), np.mean(np.abs(err))
    heavy = rows[rows[:, 0] == 420.0, 5]
    print(f"  validation: R2 = {r2:.4f}, RMSE = {rmse:.2f} mm, MAE = {mae:.2f} mm, "
          f"|relative error| <= {np.max(np.abs(rows[:, 5])):.1f} %")
    print(f"  420 g: {np.sum(heavy > 0)} of {len(heavy)} errors positive, mean {heavy.mean():.1f} %")
    with open(os.path.join(OUT, "static_validation.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mass_g", "L_cm", "experiment_mm", "simulation_mm", "error_mm", "relative_error_percent"])
        w.writerows(rows.tolist())
    with open(os.path.join(OUT, "identified_stiffness.txt"), "w", encoding="utf-8") as f:
        f.write(f"K = {k_root:.10g} N m/rad\nR2 = {r2:.6f}\nRMSE = {rmse:.4f} mm\nMAE = {mae:.4f} mm\n")
    figure(rows, r2, rmse, mae)


def figure(rows, r2, rmse, mae):
    from matplotlib.lines import Line2D
    from matplotlib.offsetbox import AnnotationBbox, DrawingArea, HPacker, TextArea
    colors = {220: "#0072B2", 320: "#009E73", 420: "#CC79A7"}
    markers = {220: "s", 320: "^", 420: "D"}
    W, H, S = 173.8, 64.0, 43.0
    left, gap, bottom = 11.6, 4.4, 10.2
    x0 = [left + i * (S + gap + left) for i in range(3)]
    fig = plotstyle.figure(W, H)
    axes = [fig.add_axes([x / W, bottom / H, S / W, S / H]) for x in x0]
    for mass in (220, 320, 420):
        g = rows[rows[:, 0] == mass]
        g = g[np.argsort(g[:, 1])]
        c, mk = colors[mass], markers[mass]
        axes[0].plot(g[:, 1], g[:, 3], color=c, lw=1.5, zorder=2)
        axes[0].plot(g[:, 1], g[:, 2], ls="none", marker=mk, ms=4.7, mfc="white", mew=1.1, mec=c, zorder=3)
        axes[1].scatter(g[:, 2], g[:, 3], s=25, marker=mk, facecolors="white", edgecolors=c, linewidths=1.1, zorder=3)
        axes[2].plot(g[:, 1], g[:, 5], color=c, marker=mk, ms=4.2, mfc="white", mew=1.0, lw=1.5, zorder=2)
    axes[0].set(xlabel=r"Deployed length $\ell$ [cm]", ylabel=r"Tip deflection $\delta_{\eta}$ [mm]", xticks=[60, 80, 100, 120])
    values = np.r_[rows[:, 2], rows[:, 3]]
    lo, hi = max(0.0, values.min() - 3.0), values.max() + 4.0
    axes[1].plot([lo, hi], [lo, hi], color="0.25", lw=1.0, ls="--", label="Perfect agreement", zorder=1)
    axes[1].set(xlim=(lo, hi), ylim=(lo, hi), xlabel=r"Experimental $\delta_{\eta}$ [mm]",
                ylabel=r"Simulated $\delta_{\eta}$ [mm]", xticks=[20, 40, 60, 80, 100], yticks=[20, 40, 60, 80, 100])
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].text(0.04, 0.96, f"$R^2$ = {r2:.4f}\nRMSE = {rmse:.2f} mm\nMAE = {mae:.2f} mm", transform=axes[1].transAxes,
                 va="top", ha="left", bbox=dict(facecolor="white", edgecolor="none", pad=0.5))
    axes[1].legend(frameon=True, facecolor="white", edgecolor="none", framealpha=1.0, loc="lower right")
    axes[2].axhline(0, color="0.25", lw=0.8, zorder=1)
    m = float(np.abs(rows[:, 5]).max())
    axes[2].set(ylim=(-m - 2.0, m + 2.0), xlabel=r"Deployed length $\ell$ [cm]", ylabel="Relative error [%]",
                xticks=[60, 80, 100, 120])
    for ax in axes:
        ax.tick_params(direction="out", width=0.6, length=2.4)
        ax.grid(True, color="0.88", linewidth=0.55, zorder=0)
    def legend_entry(label, color, marker=None):
        if marker is None:
            symbol = DrawingArea(17, 8)
            symbol.add_artist(Line2D([0.5, 16.5], [4, 4], color=color, lw=1.4, solid_capstyle="butt"))
        else:
            symbol = DrawingArea(8, 8)
            symbol.add_artist(Line2D([4], [4], ls="none", marker=marker, ms=4.6,
                                     mfc="white", mew=1.05, mec=color))
        return HPacker(children=[symbol, TextArea(label)], align="center", pad=0, sep=2.5)

    mass_entries = [TextArea("Tip mass", textprops={"weight": "bold"})]
    mass_entries.extend(legend_entry(f"{mass} g", colors[mass], markers[mass]) for mass in colors)
    mass_legend = HPacker(children=mass_entries, align="center", pad=0, sep=11)
    data_legend = HPacker(children=[legend_entry("Simulation", "0.30"),
                                   legend_entry("Experiment", "0.30", "o")],
                          align="center", pad=0, sep=11)
    shared_legend = HPacker(children=[mass_legend, data_legend], align="center", pad=0, sep=21)
    legend_y = 1 - 6.4 / (H * 72 / 25.4)
    fig.add_artist(AnnotationBbox(shared_legend, (0.5, legend_y),
                                  xycoords=fig.transFigure, frameon=False, pad=0))
    for x, label in zip(x0, ("(a)", "(b)", "(c)")):
        plotstyle.panel_label(fig, (x - left + 0.2) / W, (bottom + S + 1.4) / H, label)
    plotstyle.save(fig, paths.figure_path("static_root_stiffness_validation"))


if __name__ == "__main__":
    main()
