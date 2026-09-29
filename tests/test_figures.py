import polars as pl
from matplotlib.colors import to_hex

from mlops_core.analysis.figures import SERIES, target_distribution_figure


def test_every_period_gets_a_box_in_the_house_colours() -> None:
    """The fifth read of a catalogue fell back to matplotlib's own style."""
    reads = [f"2026-09-2{day}" for day in range(2, 7)]
    table = pl.DataFrame(
        {"snapshot": reads, "n": [5] * 5, "median": [2.0] * 5, "q25": [1.0] * 5,
         "q75": [3.0] * 5, "min": [0.0] * 5, "max": [4.0] * 5}
    )  # fmt: skip

    figure = target_distribution_figure(table, "snapshot", "price")

    faces = [to_hex(patch.get_facecolor(), keep_alpha=False) for patch in figure.axes[0].patches]
    assert len(faces) == 5 and faces[4] == SERIES[0]
