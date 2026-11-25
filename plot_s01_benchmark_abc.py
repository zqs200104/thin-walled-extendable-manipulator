"""Create a three-panel deployment benchmark figure.

Panels show the transverse tip response, deviations from the published
reference, and vibration frequency against deployed length.
"""

from pathlib import Path

from stemarm.paths import RESULTS_DIR

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    annotate,
    element_blank,
    element_line,
    element_rect,
    element_text,
    geom_hline,
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

from plot_s01_local_frequency import load_frequency_data


# -----------------------------------------------------------------------------
# Publication parameters
# -----------------------------------------------------------------------------
TEXT_SIZE_PT = 8.0
DATA_LINE_WIDTH = 0.70
REFERENCE_LINE_WIDTH = 0.78
MARKER_SIZE = 1.65
GRID_LINE_WIDTH = 0.32
AXIS_LINE_WIDTH = 0.5
WIDTH_MM = 85.0
HEIGHT_MM = 118.0
FONT_FAMILY = "Times New Roman"

# Bang Wong palette in stable series order.
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

SERIES_ORDER = ["Present model", "ALE-ANCF", "Park et al. reference"]
SERIES_COLORS = dict(zip(SERIES_ORDER, WONG_COLORS[:3]))
SERIES_LINETYPES = {
    "Present model": "solid",
    "ALE-ANCF": "dashdot",
    "Park et al. reference": "dashed",
}

ROOT = Path(__file__).resolve().parent
RESULT_DIR = Path(RESULTS_DIR) / "s01_ancf_benchmark"
FIGURE_DIR = Path(RESULTS_DIR) / "figures"
REFERENCE_CSV = ROOT / "data" / "park2013" / "fig3_tip_displacement_digitized.csv"
PRESENT_CSV = RESULT_DIR / "tip_beam_0.1_m_s.csv"
ALE_CSV = RESULT_DIR / "tip_ale_0.1_m_s.csv"
PDF_OUTPUT = FIGURE_DIR / "fig3_ANCF_ALE_validation_abc.pdf"
PNG_OUTPUT = FIGURE_DIR / "fig3_ANCF_ALE_validation_abc.png"


def publication_theme(legend_position="none"):
    return theme_bw(base_family=FONT_FAMILY, base_size=TEXT_SIZE_PT) + theme(
        text=element_text(family=FONT_FAMILY, size=TEXT_SIZE_PT, color="black"),
        axis_title=element_text(family=FONT_FAMILY, size=TEXT_SIZE_PT, color="black"),
        axis_text=element_text(family=FONT_FAMILY, size=TEXT_SIZE_PT, color="black"),
        legend_text=element_text(family=FONT_FAMILY, size=TEXT_SIZE_PT, color="black"),
        legend_title=element_blank(),
        legend_position=legend_position,
        legend_direction="horizontal",
        legend_background=element_rect(fill="none", color="none"),
        legend_key=element_rect(fill="none", color="none"),
        panel_background=element_rect(fill="white", color="none"),
        plot_background=element_rect(fill="none", color="none"),
        panel_grid_major=element_line(color="#D9D9D9", size=GRID_LINE_WIDTH),
        panel_grid_minor=element_blank(),
        axis_line=element_line(color="black", size=AXIS_LINE_WIDTH),
        axis_ticks=element_line(color="black", size=AXIS_LINE_WIDTH),
        plot_title=element_blank(),
        plot_subtitle=element_blank(),
        plot_margin=0.015,
    )


def load_response_data():
    reference = np.loadtxt(REFERENCE_CSV, delimiter=",", skiprows=1)
    present = np.loadtxt(PRESENT_CSV, delimiter=",", skiprows=1)
    ale = np.loadtxt(ALE_CSV, delimiter=",", skiprows=1)

    response = pd.concat(
        [
            pd.DataFrame(
                {"time_s": present[:, 0], "tip_mm": 1e3 * present[:, 2], "series": "Present model"}
            ),
            pd.DataFrame(
                {"time_s": ale[:, 0], "tip_mm": 1e3 * ale[:, 2], "series": "ALE-ANCF"}
            ),
            pd.DataFrame(
                {"time_s": reference[:, 0], "tip_mm": reference[:, 1], "series": "Park et al. reference"}
            ),
        ],
        ignore_index=True,
    )
    response["series"] = pd.Categorical(response["series"], categories=SERIES_ORDER, ordered=True)

    deviation = pd.concat(
        [
            pd.DataFrame(
                {
                    "time_s": reference[:, 0],
                    "deviation_mm": np.interp(reference[:, 0], present[:, 0], 1e3 * present[:, 2]) - reference[:, 1],
                    "series": "Present model",
                }
            ),
            pd.DataFrame(
                {
                    "time_s": reference[:, 0],
                    "deviation_mm": np.interp(reference[:, 0], ale[:, 0], 1e3 * ale[:, 2]) - reference[:, 1],
                    "series": "ALE-ANCF",
                }
            ),
        ],
        ignore_index=True,
    )
    deviation["series"] = pd.Categorical(deviation["series"], categories=SERIES_ORDER[:2], ordered=True)
    return response, deviation


def panel_a(response):
    return (
        ggplot(response, aes("time_s", "tip_mm", color="series", linetype="series", group="series"))
        + geom_line(size=DATA_LINE_WIDTH)
        + scale_color_manual(values=SERIES_COLORS, limits=SERIES_ORDER)
        + scale_linetype_manual(values=SERIES_LINETYPES, limits=SERIES_ORDER)
        + scale_x_continuous(breaks=np.arange(0, 10.1, 2), limits=(0, 10), expand=(0, 0))
        + scale_y_continuous(breaks=[-8, -4, 0, 4, 8, 12], limits=(-9.5, 12.5), expand=(0, 0))
        + labs(x=None, y="Tip displacement [mm]", color=None, linetype=None)
        + guides(color=guide_legend(nrow=2), linetype=guide_legend(nrow=2))
        + annotate("text", x=0.18, y=11.5, label="(a)", size=TEXT_SIZE_PT, family=FONT_FAMILY)
        + publication_theme("inside")
        + theme(
            axis_title_x=element_blank(),
            axis_text_x=element_blank(),
            axis_ticks_major_x=element_blank(),
            legend_position_inside=(0.72, 0.93),
            legend_background=element_rect(fill="white", color="none"),
        )
    )


def panel_b(deviation):
    limit = 1.12 * float(np.max(np.abs(deviation["deviation_mm"])))
    return (
        ggplot(deviation, aes("time_s", "deviation_mm", color="series", linetype="series", group="series"))
        + geom_hline(yintercept=0.0, color="#777777", size=0.4)
        + geom_line(size=DATA_LINE_WIDTH)
        + scale_color_manual(values=SERIES_COLORS, limits=SERIES_ORDER[:2])
        + scale_linetype_manual(values=SERIES_LINETYPES, limits=SERIES_ORDER[:2])
        + scale_x_continuous(breaks=np.arange(0, 10.1, 2), limits=(0, 10), expand=(0, 0))
        + scale_y_continuous(limits=(-limit, limit), expand=(0, 0))
        + labs(x="Time [s]", y="Deviation [mm]", color=None, linetype=None)
        + annotate("text", x=0.18, y=0.88 * limit, label="(b)", size=TEXT_SIZE_PT, family=FONT_FAMILY)
        + publication_theme("none")
    )


def panel_c(frequency):
    frequency = frequency.copy()
    frequency["series"] = frequency["series"].astype(str).replace(
        {"Park et al. reference": "Park et al. reference"}
    )
    frequency["series"] = pd.Categorical(
        frequency["series"], categories=["Present model", "Park et al. reference"], ordered=True
    )
    shapes = {"Present model": "o", "Park et al. reference": "s"}
    return (
        ggplot(
            frequency,
            aes(
                "deployed_length_m",
                "local_frequency_hz",
                color="series",
                linetype="series",
                shape="series",
                group="series",
            ),
        )
        + geom_line(size=DATA_LINE_WIDTH)
        + geom_point(size=MARKER_SIZE, stroke=0.4, fill="white")
        + scale_color_manual(values=SERIES_COLORS, limits=["Present model", "Park et al. reference"])
        + scale_linetype_manual(values=SERIES_LINETYPES, limits=["Present model", "Park et al. reference"])
        + scale_shape_manual(values=shapes, limits=["Present model", "Park et al. reference"])
        + scale_x_continuous(breaks=np.arange(1.8, 2.81, 0.2), limits=(1.78, 2.8), expand=(0, 0))
        + scale_y_continuous(breaks=np.arange(1.0, 2.31, 0.2), limits=(1.0, 2.35), expand=(0, 0))
        + labs(x="Deployed length [m]", y="Vibration frequency [Hz]", color=None, linetype=None, shape=None)
        + annotate("text", x=2.76, y=2.31, label="(c)", size=TEXT_SIZE_PT, family=FONT_FAMILY)
        + publication_theme("none")
    )


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    response, deviation = load_response_data()
    frequency = load_frequency_data()
    width_in = WIDTH_MM / 25.4
    height_in = HEIGHT_MM / 25.4
    figure = panel_a(response) / panel_b(deviation) / panel_c(frequency)
    figure.save(PDF_OUTPUT, width=width_in, height=height_in, units="in", verbose=False)
    figure = panel_a(response) / panel_b(deviation) / panel_c(frequency)
    figure.save(PNG_OUTPUT, width=width_in, height=height_in, units="in", dpi=300, verbose=False)
    print(f"figure: {PDF_OUTPUT}")
    print(f"preview: {PNG_OUTPUT}")


if __name__ == "__main__":
    main()
