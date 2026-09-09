"""The opt-in committee uncertainty figure (energy + fmax, each with a band).

Separate from ``test_committee_plot.py``, which covers the sigma panel on the
convergence figure. This one is the standalone ``*_uncertainty.png``: mean
energy and its uncertainty band on the primary y-axis, max force and its
uncertainty band on the secondary.
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pytest
from ase.build import bulk
from ase.io import write

matplotlib.use("Agg")

from matplotlib import pyplot as plt                # noqa: E402
from typer.testing import CliRunner                 # noqa: E402

from mliprun.cli.commands.optimize import app       # noqa: E402
from mliprun.core.committee.calculator import CommitteeCalculator  # noqa: E402
from mliprun.core.committee.remote import RemoteMember             # noqa: E402
from mliprun.core.optimize import (                 # noqa: E402
    _plot_uncertainty,
    _uncertainty_traces,
    run_optimization,
)

runner = CliRunner()


def _rows(energies, spreads, fmax_values, sigmas):
    """A committee trace carrying exactly the columns the figure reads."""
    return [
        {"step": i,
         "energy_mean_eV": energy,
         "energy_spread_aligned_eV": spread,
         "fmax_eV_per_A": fmax_value,
         "sigma_max_free_eV_per_A": sigma}
        for i, (energy, spread, fmax_value, sigma)
        in enumerate(zip(energies, spreads, fmax_values, sigmas))
    ]


def _default_rows():
    """Three steps of a plausible committee relaxation.

    The step-0 spread is 0.0 because that is the row each member's own energy
    offset is measured against, which is exactly the artefact the figure has
    to annotate rather than hide.
    """
    return _rows(energies=[-88.10, -88.31, -88.42],
                 spreads=[0.0, 0.012, 0.030],
                 fmax_values=[1.80, 0.54, 0.02],
                 sigmas=[0.21, 0.09, 0.04])


def _rows_with_no_clipping():
    """sigma below fmax at every step, so no band edge reaches zero."""
    return _rows(energies=[-88.10, -88.31, -88.42],
                 spreads=[0.0, 0.012, 0.030],
                 fmax_values=[1.80, 0.54, 0.20],
                 sigmas=[0.21, 0.09, 0.04])


def _band_at_each_step(collection, steps):
    """Recover ``(lower, upper)`` per x from a ``fill_between`` polygon.

    Read back by x value rather than by vertex position: the order matplotlib
    builds the polygon in is an implementation detail, but under any ordering
    the lowest and highest y at a given x are that step's band edges.
    """
    vertices = collection.get_paths()[0].vertices
    lower, upper = [], []
    for step in steps:
        ys = vertices[np.isclose(vertices[:, 0], float(step)), 1]
        lower.append(float(ys.min()))
        upper.append(float(ys.max()))
    return lower, upper


def _emt_committee(tmp_path, names=("member_a", "member_b")):
    """Two reserved-``emt`` members, so no MLIP is needed."""
    members = [
        RemoteMember(name, sys.executable, mlip="emt",
                     log_path=tmp_path / f"committee_{name}.log",
                     timeout=60.0)
        for name in names
    ]
    return CommitteeCalculator(members, mixed_theory=False,
                               levels=("EMT", "EMT"))


def _rattled():
    atoms = bulk("Cu", "fcc", a=3.6) * (2, 2, 1)
    atoms.rattle(stdev=0.05, seed=7)
    return atoms


class TestTheNumbers:
    """``_uncertainty_traces`` -- the arithmetic, with no matplotlib in it."""

    def test_the_energy_is_relative_to_step_zero(self):
        traces = _uncertainty_traces(_default_rows())
        assert traces["energy"] == pytest.approx(
            [0.0, -0.21, -0.32], abs=1e-12)
        # Exactly zero, not approximately: the band shares this zero, so a
        # drifting reference would make centre and band incommensurate.
        assert traces["energy"][0] == 0.0

    def test_the_energy_band_is_the_aligned_spread(self):
        traces = _uncertainty_traces(_default_rows())
        assert traces["energy_lo"] == pytest.approx(
            [0.0, -0.222, -0.350], abs=1e-12)
        assert traces["energy_hi"] == pytest.approx(
            [0.0, -0.198, -0.290], abs=1e-12)

    def test_the_step_zero_energy_band_has_zero_width(self):
        """Zero by construction, and the figure must not paper over it."""
        traces = _uncertainty_traces(_default_rows())
        assert traces["energy_hi"][0] - traces["energy_lo"][0] == 0.0

    def test_the_force_band_is_fmax_plus_minus_sigma_max_free(self):
        traces = _uncertainty_traces(_default_rows())
        assert traces["fmax"] == pytest.approx([1.80, 0.54, 0.02], abs=1e-12)
        assert traces["force_lo"] == pytest.approx(
            [1.59, 0.45, 0.0], abs=1e-12)
        assert traces["force_hi"] == pytest.approx(
            [2.01, 0.63, 0.06], abs=1e-12)

    def test_a_sigma_larger_than_fmax_clips_the_lower_edge_at_zero(self):
        """A force magnitude cannot be negative.

        sigma above fmax is the interesting regime -- the members disagree
        about the force by more than its own size, so the minimum sits inside
        the committee's noise -- and it must render as a band reaching the
        axis floor, never as a negative force.
        """
        rows = _rows(energies=[-1.0, -1.1], spreads=[0.0, 0.01],
                     fmax_values=[0.05, 0.03], sigmas=[0.20, 0.15])
        traces = _uncertainty_traces(rows)
        assert traces["force_lo"] == pytest.approx([0.0, 0.0], abs=1e-12)
        assert traces["force_hi"] == pytest.approx([0.25, 0.18], abs=1e-12)


class TestTheFigure:
    """``_plot_uncertainty`` -- what actually gets drawn."""

    def test_the_energy_is_on_the_left_and_the_force_on_the_right(self):
        figure = _plot_uncertainty(_default_rows(), 0.05)
        energy_axis, force_axis = figure.axes[0], figure.axes[1]
        assert energy_axis.yaxis.get_label_position() == "left"
        assert force_axis.yaxis.get_label_position() == "right"
        assert "eV" in energy_axis.get_ylabel()
        assert "eV/Ang" in force_axis.get_ylabel()
        plt.close(figure)

    def test_the_two_axes_share_the_x_axis(self):
        """One step axis, not two that drift apart under autoscaling."""
        figure = _plot_uncertainty(_default_rows(), 0.05)
        energy_axis, force_axis = figure.axes[0], figure.axes[1]
        assert energy_axis.get_shared_x_axes().joined(energy_axis, force_axis)
        plt.close(figure)

    def test_the_energy_trace_is_drawn_relative_to_step_zero(self):
        figure = _plot_uncertainty(_default_rows(), 0.05)
        line = figure.axes[0].lines[0]
        assert list(line.get_xdata()) == pytest.approx([0, 1, 2], abs=1e-12)
        assert list(line.get_ydata()) == pytest.approx(
            [0.0, -0.21, -0.32], abs=1e-12)
        plt.close(figure)

    def test_the_force_trace_is_fmax(self):
        figure = _plot_uncertainty(_default_rows(), 0.05)
        line = figure.axes[1].lines[0]
        assert list(line.get_ydata()) == pytest.approx(
            [1.80, 0.54, 0.02], abs=1e-12)
        plt.close(figure)

    def test_the_drawn_energy_band_matches_the_computed_one(self):
        rows = _default_rows()
        traces = _uncertainty_traces(rows)
        figure = _plot_uncertainty(rows, 0.05)

        assert len(figure.axes[0].collections) == 1
        lower, upper = _band_at_each_step(figure.axes[0].collections[0],
                                          traces["steps"])
        assert lower == pytest.approx(traces["energy_lo"], abs=1e-12)
        assert upper == pytest.approx(traces["energy_hi"], abs=1e-12)
        plt.close(figure)

    def test_the_drawn_force_band_matches_the_computed_one(self):
        rows = _default_rows()
        traces = _uncertainty_traces(rows)
        figure = _plot_uncertainty(rows, 0.05)

        assert len(figure.axes[1].collections) == 1
        lower, upper = _band_at_each_step(figure.axes[1].collections[0],
                                          traces["steps"])
        assert lower == pytest.approx(traces["force_lo"], abs=1e-12)
        assert upper == pytest.approx(traces["force_hi"], abs=1e-12)
        plt.close(figure)

    def test_the_clipped_force_band_is_drawn_at_zero_not_below(self):
        rows = _rows(energies=[-1.0, -1.1], spreads=[0.0, 0.01],
                     fmax_values=[0.05, 0.03], sigmas=[0.20, 0.15])
        figure = _plot_uncertainty(rows, 0.05)
        lower, _ = _band_at_each_step(figure.axes[1].collections[0], [0, 1])
        assert lower == pytest.approx([0.0, 0.0], abs=1e-12)
        plt.close(figure)

    def test_the_energy_axis_is_linear(self):
        """Energy relative to step 0 is negative for any relaxation that
        went downhill, and a log axis cannot carry a negative number."""
        figure = _plot_uncertainty(_default_rows(), 0.05)
        assert figure.axes[0].get_yscale() == "linear"
        plt.close(figure)

    def test_the_force_axis_is_log_when_no_band_edge_reaches_zero(self):
        """A relaxation spans orders of magnitude in fmax; linear buries
        everything after the first few steps."""
        figure = _plot_uncertainty(_rows_with_no_clipping(), 0.05)
        assert figure.axes[1].get_yscale() == "log"
        plt.close(figure)

    def test_a_band_edge_clipped_to_zero_forces_symlog(self):
        """A log axis cannot draw zero, and matplotlib drops the offending
        vertices rather than warning -- which silently deforms the band
        polygon exactly where sigma exceeds fmax. symlog keeps the edge.
        """
        rows = _default_rows()          # last step: 0.02 - 0.04 -> clipped
        traces = _uncertainty_traces(rows)
        assert min(traces["force_lo"]) == 0.0, "fixture must clip somewhere"

        figure = _plot_uncertainty(rows, 0.05)
        force_axis = figure.axes[1]
        assert force_axis.get_yscale() == "symlog"

        smallest_positive = min(v for v in (traces["fmax"]
                                            + traces["force_lo"]
                                            + traces["force_hi"]) if v > 0)
        assert force_axis.yaxis._scale.linthresh == pytest.approx(
            smallest_positive)

        # Every point survives, the clipped edge included: this is the
        # assertion that proves nothing was silently dropped.
        lower, upper = _band_at_each_step(force_axis.collections[0],
                                          traces["steps"])
        assert lower == pytest.approx(traces["force_lo"], abs=1e-12)
        assert upper == pytest.approx(traces["force_hi"], abs=1e-12)
        plt.close(figure)

    def test_the_force_axis_never_shows_a_negative_force(self):
        """symlog is symmetric about zero by definition, and autoscaling it
        offers decades of negative force -- a quantity that does not exist."""
        figure = _plot_uncertainty(_default_rows(), 0.05)
        bottom, _ = figure.axes[1].get_ylim()
        assert bottom == 0.0
        plt.close(figure)

    def test_an_all_zero_force_trace_stays_linear(self):
        """A converged, exactly-agreeing committee has nothing positive to
        put on a log axis; linear keeps the flat trace on screen."""
        rows = _rows(energies=[-1.0, -1.0], spreads=[0.0, 0.0],
                     fmax_values=[0.0, 0.0], sigmas=[0.0, 0.0])
        figure = _plot_uncertainty(rows, 0.05)
        assert figure.axes[1].get_yscale() == "linear"
        plt.close(figure)

    def test_the_fmax_target_line_sits_on_the_force_axis(self):
        figure = _plot_uncertainty(_default_rows(), 0.05)
        target_lines = [line for line in figure.axes[1].lines
                        if list(line.get_ydata()) == pytest.approx(
                            [0.05, 0.05], abs=1e-12)]
        assert len(target_lines) == 1
        assert target_lines[0].get_linestyle() == "--"
        plt.close(figure)

    def test_the_step_zero_artefact_is_annotated(self):
        """Without the note, a band opening from zero reads as a run that
        started certain and got worse, which is not what happened."""
        figure = _plot_uncertainty(_default_rows(), 0.05)
        texts = ([text.get_text() for text in figure.texts]
                 + [text.get_text() for axis in figure.axes
                    for text in axis.texts])
        assert any("step 0" in text and "construction" in text
                   for text in texts)
        plt.close(figure)

    def test_a_single_step_trace_draws_nothing(self):
        """A run converged at step 0 has one row; a one-point figure with a
        zero-width band would say nothing and imply a lot."""
        assert _plot_uncertainty(_default_rows()[:1], 0.05) is None

    def test_an_empty_trace_draws_nothing(self):
        assert _plot_uncertainty([], 0.05) is None
        assert _plot_uncertainty(None, 0.05) is None


class TestRunOptimizationWiring:
    """The figure reaches disk only when asked for."""

    def test_no_figure_without_the_flag(self, tmp_path):
        atoms = _rattled()
        committee = _emt_committee(tmp_path)
        committee.start()
        atoms.calc = committee
        try:
            run_optimization(atoms, fmax=0.001, max_steps=5,
                             output_dir=tmp_path, model_name="committee",
                             verbose=False, committee=committee)
        finally:
            committee.close()

        assert (tmp_path / "opt_committee.csv").exists()
        assert not (tmp_path / "opt_uncertainty.png").exists()

    def test_the_figure_is_written_with_the_flag(self, tmp_path):
        atoms = _rattled()
        committee = _emt_committee(tmp_path)
        committee.start()
        atoms.calc = committee
        try:
            run_optimization(atoms, fmax=0.001, max_steps=5,
                             output_dir=tmp_path, model_name="committee",
                             verbose=False, committee=committee,
                             uncertainty_plot=True)
        finally:
            committee.close()

        figure_path = tmp_path / "opt_uncertainty.png"
        assert figure_path.exists()
        assert figure_path.stat().st_size > 0

    def test_the_filename_follows_the_logfile(self, tmp_path):
        atoms = _rattled()
        committee = _emt_committee(tmp_path)
        committee.start()
        atoms.calc = committee
        try:
            run_optimization(atoms, fmax=0.001, max_steps=5,
                             output_dir=tmp_path, model_name="committee",
                             verbose=False, committee=committee,
                             logfile="scan7.log", trajectory="scan7.traj",
                             uncertainty_plot=True)
        finally:
            committee.close()

        assert (tmp_path / "scan7_uncertainty.png").exists()
        assert not (tmp_path / "opt_uncertainty.png").exists()

    def test_a_run_converged_at_step_zero_writes_nothing_and_says_so(
            self, tmp_path, caplog):
        """A perfect lattice has zero forces, so the run converges at step 0
        and the trace has a single row. Silence there would look like a bug
        in the flag; the run has to say why there is no figure."""
        atoms = bulk("Cu", "fcc", a=3.6) * (2, 2, 1)   # unrattled: F = 0
        committee = _emt_committee(tmp_path)
        committee.start()
        atoms.calc = committee
        try:
            with caplog.at_level("INFO", logger="mliprun.core.optimize"):
                run_optimization(atoms, fmax=0.05, max_steps=5,
                                 output_dir=tmp_path, model_name="committee",
                                 verbose=False, committee=committee,
                                 uncertainty_plot=True)
        finally:
            committee.close()

        assert not (tmp_path / "opt_uncertainty.png").exists()
        assert any("uncertainty plot skipped" in record.message
                   for record in caplog.records)

    def test_without_a_committee_the_flag_writes_nothing(self, tmp_path):
        """The API has no committee to disagree; it must not crash either."""
        from ase.calculators.emt import EMT

        atoms = _rattled()
        atoms.calc = EMT()
        run_optimization(atoms, fmax=0.001, max_steps=5,
                         output_dir=tmp_path, model_name="emt",
                         verbose=False, uncertainty_plot=True)

        assert (tmp_path / "opt_convergence.csv").exists()
        assert not (tmp_path / "opt_uncertainty.png").exists()


class TestTheCliFlag:
    @pytest.fixture
    def structure(self, tmp_path):
        path = tmp_path / "POSCAR"
        write(str(path), _rattled(), format="vasp")
        return path

    def test_the_flag_without_a_committee_is_rejected(self, structure,
                                                      monkeypatch):
        """Silently doing nothing would leave the user waiting for a figure
        that was never going to appear."""
        import mliprun.cli.commands.optimize as optimize_cli

        monkeypatch.setattr(optimize_cli, "validate_mlip", lambda *a, **k: None)
        result = runner.invoke(app, ["run", "--structure", str(structure),
                                     "--mlip", "emt", "--uncertainty-plot"])
        assert result.exit_code == 1
        assert "--uncertainty-plot" in result.output
        assert "--committee" in result.output

    def test_a_committee_run_writes_and_lists_the_figure(
            self, structure, fake_committee_file, tmp_path):
        path, _ = fake_committee_file
        result = runner.invoke(app, ["run", "--structure", str(structure),
                                     "--committee", str(path),
                                     "--fmax", "0.001", "--max-steps", "5",
                                     "--no-verbose", "--uncertainty-plot"])
        assert result.exit_code == 0, result.output

        figure_path = structure.parent / "opt_uncertainty.png"
        assert figure_path.exists()
        assert "opt_uncertainty.png" in result.output

    def test_no_figure_when_the_flag_is_absent(self, structure,
                                               fake_committee_file):
        path, _ = fake_committee_file
        result = runner.invoke(app, ["run", "--structure", str(structure),
                                     "--committee", str(path),
                                     "--fmax", "0.001", "--max-steps", "5",
                                     "--no-verbose"])
        assert result.exit_code == 0, result.output
        assert not (structure.parent / "opt_uncertainty.png").exists()
        assert "opt_uncertainty.png" not in result.output
