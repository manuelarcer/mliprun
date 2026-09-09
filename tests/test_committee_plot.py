"""The optional sigma panel on the convergence figure."""
import warnings

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
    return [{"step": i, "sigma_max_free_eV_per_A": s,
             "sigma_mean_free_eV_per_A": s / 2.0}
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


def test_all_zero_sigma_uses_linear_scale_and_annotates(convergence_frame):
    """Exact committee agreement at every step (sigma identically 0).

    A plain log scale would silently drop every point with no warning,
    leaving an apparently empty panel. The fix must keep both flat-zero
    traces genuinely on screen and explain why the panel is flat.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        figure = _plot_convergence(convergence_frame, 0.05, "bfgs",
                                   committee_rows=_rows([0.0, 0.0, 0.0]))
        figure.canvas.draw()

    # No warning masks the defect either way -- the bug is silent, and so
    # is the fix. The real proof is the rendering check below.
    assert caught == []

    sigma_axis = figure.axes[2]
    assert sigma_axis.get_yscale() == "linear"

    sigma_max_line, sigma_mean_line = sigma_axis.lines[0], sigma_axis.lines[1]
    assert list(sigma_max_line.get_ydata()) == pytest.approx(
        [0.0, 0.0, 0.0], abs=1e-12)
    assert list(sigma_mean_line.get_ydata()) == pytest.approx(
        [0.0, 0.0, 0.0], abs=1e-12)

    # Genuinely rendered, not just present in the data: on the broken log
    # scale, autoscale ignores the non-positive sigma values entirely and
    # the view limits sit off the fmax reference line alone (e.g.
    # ~0.009-0.11), pushing both zero traces below the visible range. A
    # linear axis around all-zero data keeps 0 inside view.
    ymin, ymax = sigma_axis.get_ylim()
    assert ymin <= 0.0 <= ymax

    texts = [t.get_text() for t in sigma_axis.texts]
    assert any("sigma is identically zero" in t and "agree exactly" in t
               for t in texts)

    plt.close(figure)


def test_mixed_zero_and_positive_sigma_uses_symlog_and_keeps_every_point(
        convergence_frame):
    """Some steps agree exactly, others show real disagreement.

    A log scale would drop precisely the exact-agreement step while
    plotting the rest normally -- more misleading than an empty panel,
    since it looks like clean data. symlog must keep every point.
    """
    sigma_max_values = [0.0, 0.02, 0.9]
    sigma_mean_values = [v / 2.0 for v in sigma_max_values]  # 0.0, 0.01, 0.45
    figure = _plot_convergence(convergence_frame, 0.05, "bfgs",
                               committee_rows=_rows(sigma_max_values))

    sigma_axis = figure.axes[2]
    assert sigma_axis.get_yscale() == "symlog"

    smallest_positive = min(v for v in sigma_max_values + sigma_mean_values
                             if v > 0)
    assert sigma_axis.yaxis._scale.linthresh == pytest.approx(
        smallest_positive)

    sigma_max_line, sigma_mean_line = sigma_axis.lines[0], sigma_axis.lines[1]
    # Every point survives, including the exact-zero step -- this is the
    # assertion that proves nothing was silently dropped.
    assert list(sigma_max_line.get_ydata()) == pytest.approx(
        sigma_max_values, abs=1e-12)
    assert list(sigma_mean_line.get_ydata()) == pytest.approx(
        sigma_mean_values, abs=1e-12)

    plt.close(figure)
