"""Plot local vibration frequency against deployed length for the Park benchmark.

Local frequency at an internal zero crossing is computed from the preceding and
following zero crossings of the transverse tip displacement. These crossings
span one complete cycle, and the deployed length follows L(t) = 1.8 + 0.1 t.
"""

from pathlib import Path

from stemarm.paths import RESULTS_DIR

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    element_blank,
    element_line,
    element_rect,
    element_text,
    geom_line,
    geom_point,
    ggplot,
    guides,
    guide_legend,
    labs,
    scale_color_manual,
    scale_linetype_manual,
    scale_shape_manual,
    scale_x_continuous,
    scale_y_continuous,
    theme,
    theme_bw,
)


# -----------------------------------------------------------------------------
# Publication parameters
# -----------------------------------------------------------------------------
TEXT_SIZE_PT = 8.0
DATA_LINE_WIDTH = 0.75
MARKER_SIZE = 2.0
GRID_LINE_WIDTH = 0.35
AXIS_LINE_WIDTH = 0.5
WIDTH_MM = 85.0
HEIGHT_MM = 58.0
FONT_FAMILY = "Times New Roman"

# Bang Wong palette in the required stable order.
WONG_COLORS = [
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#CC79A7",
    "#D55E00",
    "#F0E442",
    "#0072B2",
    "#000000",
]

# Reserved for continuous color encodings. When used, finite observations are
# split at equal-count quantiles and mapped through the full viridis ramp.
CONTINUOUS_COLOR_MAP = "viridis"
CONTINUOUS_COLOR_RULE = "equal-count quantiles from finite values"

INITIAL_LENGTH_M = 1.8
DEPLOYMENT_SPEED_M_S = 0.1
SERIES_ORDER = ["Park et al. reference", "Present model"]

ROOT = Path(__file__).resolve().parent
REFERENCE_CSV = ROOT / "data" / "park2013" / "fig3_tip_displacement_digitized.csv"
MODEL_CSV = Path(RESULTS_DIR) / "s01_ancf_benchmark" / "tip_beam_0.1_m_s.csv"
OUTPUT_DIR = Path(RESULTS_DIR) / "figures"
DATA_OUTPUT = Path(RESULTS_DIR) / "s01_ancf_benchmark" / "local_frequency_vs_length.csv"
PDF_OUTPUT = OUTPUT_DIR / "fig3_local_frequency_vs_length.pdf"
PNG_OUTPUT = OUTPUT_DIR / "fig3_local_frequency_vs_length.png"


def zero_crossings(time_s, displacement):
    """Return linearly interpolated zero-crossing times in both directions."""
    time_s = np.asarray(time_s, dtype=float)
    displacement = np.asarray(displacement, dtype=float)
    indices = np.flatnonzero(displacement[:-1] * displacement[1:] < 0.0)
    x0 = displacement[indices]
    x1 = displacement[indices + 1]
    t0 = time_s[indices]
    t1 = time_s[indices + 1]
    return t0 - x0 * (t1 - t0) / (x1 - x0)


def local_frequency_table(time_s, displacement, series):
    """Compute local frequency over the full cycle around each internal crossing."""
    crossings = zero_crossings(time_s, displacement)
    if len(crossings) < 3:
        raise ValueError(f"Fewer than three zero crossings for {series}")
    start = crossings[:-2]
    center = crossings[1:-1]
    end = crossings[2:]
    frequency = 1.0 / (end - start)
    deployed_length = INITIAL_LENGTH_M + DEPLOYMENT_SPEED_M_S * center
    return pd.DataFrame(
        {
            "series": series,
            "deployed_length_m": deployed_length,
            "local_frequency_hz": frequency,
            "center_crossing_time_s": center,
            "crossing_start_s": start,
            "crossing_end_s": end,
        }
    )


def load_frequency_data():
    reference = np.loadtxt(REFERENCE_CSV, delimiter=",", skiprows=1)
    model = np.loadtxt(MODEL_CSV, delimiter=",", skiprows=1)
    tables = [
        local_frequency_table(
            reference[:, 0], reference[:, 1], "Park et al. reference"
        ),
        local_frequency_table(model[:, 0], model[:, 2], "Present model"),
    ]
    data = pd.concat(tables, ignore_index=True)
    data["series"] = pd.Categorical(
        data["series"], categories=SERIES_ORDER, ordered=True
    )
    return data


def build_plot(data):
    colors = dict(zip(SERIES_ORDER, WONG_COLORS[: len(SERIES_ORDER)]))
    linetypes = {
        "Park et al. reference": "dashed",
        "Present model": "solid",
    }
    shapes = {
        "Park et al. reference": "s",
        "Present model": "o",
    }
    return (
        ggplot(
            data,
            aes(
                x="deployed_length_m",
                y="local_frequency_hz",
                color="series",
                linetype="series",
                shape="series",
                group="series",
            ),
        )
        + geom_line(size=DATA_LINE_WIDTH)
        + geom_point(size=MARKER_SIZE, stroke=0.45, fill="white")
        + scale_color_manual(values=colors)
        + scale_linetype_manual(values=linetypes)
        + scale_shape_manual(values=shapes)
        + scale_x_continuous(
            breaks=np.arange(1.8, 2.81, 0.2), limits=(1.8, 2.8), expand=(0, 0)
        )
        + scale_y_continuous(
            breaks=np.arange(1.0, 2.31, 0.2), limits=(1.0, 2.35), expand=(0, 0)
        )
        + labs(
            x="Deployed length [m]",
            y="Vibration frequency [Hz]",
            color=None,
            linetype=None,
            shape=None,
        )
        + guides(
            color=guide_legend(nrow=1),
            linetype=guide_legend(nrow=1),
            shape=guide_legend(nrow=1),
        )
        + theme_bw(base_family=FONT_FAMILY, base_size=TEXT_SIZE_PT)
        + theme(
            text=element_text(
                family=FONT_FAMILY, size=TEXT_SIZE_PT, color="black"
            ),
            axis_title=element_text(
                family=FONT_FAMILY, size=TEXT_SIZE_PT, color="black"
            ),
            axis_text=element_text(
                family=FONT_FAMILY, size=TEXT_SIZE_PT, color="black"
            ),
            legend_text=element_text(
                family=FONT_FAMILY, size=TEXT_SIZE_PT, color="black"
            ),
            legend_title=element_blank(),
            legend_position="top",
            legend_direction="horizontal",
            legend_background=element_rect(fill="none", color="none"),
            legend_key=element_rect(fill="none", color="none"),
            panel_background=element_rect(fill="white", color="none"),
            plot_background=element_rect(fill="none", color="none"),
            panel_grid_major=element_line(
                color="#D9D9D9", size=GRID_LINE_WIDTH
            ),
            panel_grid_minor=element_blank(),
            axis_line=element_line(color="black", size=AXIS_LINE_WIDTH),
            axis_ticks=element_line(color="black", size=AXIS_LINE_WIDTH),
            plot_title=element_blank(),
            plot_subtitle=element_blank(),
        )
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    data = load_frequency_data()
    data.to_csv(DATA_OUTPUT, index=False, float_format="%.9g")
    plot = build_plot(data)
    width_in = WIDTH_MM / 25.4
    height_in = HEIGHT_MM / 25.4
    plot.save(PDF_OUTPUT, width=width_in, height=height_in, units="in", verbose=False)
    plot.save(
        PNG_OUTPUT,
        width=width_in,
        height=height_in,
        units="in",
        dpi=300,
        verbose=False,
    )

    reference = data[data["series"] == "Park et al. reference"]
    model = data[data["series"] == "Present model"]
    model_at_reference = np.interp(
        reference["deployed_length_m"],
        model["deployed_length_m"],
        model["local_frequency_hz"],
    )
    relative_error = np.abs(
        model_at_reference - reference["local_frequency_hz"].to_numpy()
    ) / reference["local_frequency_hz"].to_numpy()
    print(f"data: {DATA_OUTPUT}")
    print(f"figure: {PDF_OUTPUT}")
    print(f"preview: {PNG_OUTPUT}")
    print(
        "reference frequency range: "
        f"{reference['local_frequency_hz'].iloc[0]:.3f} to "
        f"{reference['local_frequency_hz'].iloc[-1]:.3f} Hz"
    )
    print(
        "present-model frequency range: "
        f"{model['local_frequency_hz'].iloc[0]:.3f} to "
        f"{model['local_frequency_hz'].iloc[-1]:.3f} Hz"
    )
    print(f"maximum interpolated relative difference: {100 * relative_error.max():.3f}%")


if __name__ == "__main__":
    main()
