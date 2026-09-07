"""The optional sigma panel on the convergence figure."""
import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from matplotlib import pyplot as plt          # noqa: E402 -- after use("Agg")

from mliprun.core.optimize import _plot_convergence  # noqa: E402


@pytest.fixture
def convergence_frame():
    return pd.DataFrame({"step": [0, 1, 2],
                         "energy(eV)": [-1.0, -1.2, -1.3],
                         "fmax(eV/A)": [0.9, 0.3, 0.02]})


def _rows(sigmas):
    return [{"step": i, "sigma_max_eV_per_A": s,
             "sigma_mean_eV_per_A": s / 2.0}
            for i, s in enumerate(sigmas)]


def test_without_a_committee_the_figure_has_two_panels(convergence_frame):
    figure = _plot_convergence(convergence_frame, 0.05, "bfgs")
    assert len(figure.axes) == 2
    plt.close(figure)


def test_with_a_committee_a_third_panel_carries_sigma(convergence_frame):
    figure = _plot_convergence(convergence_frame, 0.05, "bfgs",
                               committee_rows=_rows([0.9, 0.3, 0.02]))
    assert len(figure.axes) == 3
    sigma_axis = figure.axes[2]
    plotted = sigma_axis.lines[0].get_ydata()
    assert list(plotted) == pytest.approx([0.9, 0.3, 0.02], abs=1e-12)
    assert sigma_axis.get_yscale() == "log"
    plt.close(figure)


def test_an_empty_committee_trace_falls_back_to_two_panels(convergence_frame):
    figure = _plot_convergence(convergence_frame, 0.05, "bfgs",
                               committee_rows=[])
    assert len(figure.axes) == 2
    plt.close(figure)
