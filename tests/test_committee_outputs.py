"""Committee outputs: the step trace, the per-atom file, and the flag."""
import csv

import numpy as np
import pytest

from mliprun.core.committee.calculator import (
    CommitteeTraceWriter,
    uncertainty_summary,
    write_peratom_sigma,
)


def _latest(energies, sigma_per_atom, energy_mean=None):
    # No mask is passed in by any caller of this helper, so it mirrors
    # committee_statistics's own unmasked default: the "_all" numbers equal
    # the headline ones and every atom counts as free.
    sigma = np.asarray(sigma_per_atom, dtype=float)
    worst = int(np.argmax(sigma))
    values = list(energies.values())
    return {
        "energies": dict(energies),
        "energy_mean": (energy_mean if energy_mean is not None
                        else float(np.mean(values))),
        "sigma_per_atom": sigma,
        "sigma_max": float(sigma[worst]),
        "sigma_mean": float(sigma.mean()),
        "worst_atom": worst,
        "sigma_max_all": float(sigma[worst]),
        "n_free_atoms": int(sigma.size),
    }


def _read(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class TestTraceWriter:
    def test_header_carries_one_column_per_member(self, tmp_path):
        path = tmp_path / "opt_committee.csv"
        writer = CommitteeTraceWriter(path, ["member_a", "member_b"], False)
        writer.write_step(0, _latest({"member_a": -1.0, "member_b": -3.0},
                                     [0.1, 0.2]), 0.5)
        writer.close()

        with open(path, newline="", encoding="utf-8") as handle:
            header = next(csv.reader(handle))
        assert header == [
            "step", "energy_mean_eV", "energy_spread_aligned_eV",
            "E_member_a_eV", "E_member_b_eV", "fmax_eV_per_A",
            "sigma_max_eV_per_A", "sigma_mean_eV_per_A", "worst_atom",
            "sigma_max_all_eV_per_A", "n_free_atoms", "mixed_theory",
        ]

    def test_values_land_in_the_right_columns(self, tmp_path):
        path = tmp_path / "opt_committee.csv"
        writer = CommitteeTraceWriter(path, ["member_a", "member_b"], True)
        writer.write_step(3, _latest({"member_a": -1.0, "member_b": -3.0},
                                     [0.1, 0.4]), 0.25)
        writer.close()

        row = _read(path)[0]
        assert int(row["step"]) == 3
        assert float(row["energy_mean_eV"]) == pytest.approx(-2.0, abs=1e-12)
        assert float(row["E_member_a_eV"]) == pytest.approx(-1.0, abs=1e-12)
        assert float(row["E_member_b_eV"]) == pytest.approx(-3.0, abs=1e-12)
        assert float(row["fmax_eV_per_A"]) == pytest.approx(0.25, abs=1e-12)
        assert float(row["sigma_max_eV_per_A"]) == pytest.approx(0.4, abs=1e-12)
        assert float(row["sigma_mean_eV_per_A"]) == pytest.approx(0.25,
                                                                  abs=1e-12)
        assert int(row["worst_atom"]) == 1
        assert row["mixed_theory"] == "True"

    def test_first_step_has_zero_aligned_spread_by_construction(self, tmp_path):
        """Step 0 defines each member's baseline, so its aligned spread is
        exactly zero -- not a measurement."""
        writer = CommitteeTraceWriter(tmp_path / "c.csv",
                                      ["member_a", "member_b"], False)
        row = writer.write_step(0, _latest({"member_a": -73.68,
                                            "member_b": -82.86}, [0.1]), 1.0)
        writer.close()
        assert row["energy_spread_aligned_eV"] == pytest.approx(0.0, abs=1e-12)

    def test_constant_offsets_cancel_on_later_steps(self, tmp_path):
        """A 9.18 eV raw spread that is pure per-model offset must report as
        zero uncertainty once both members move by the same amount."""
        writer = CommitteeTraceWriter(tmp_path / "c.csv",
                                      ["member_a", "member_b"], False)
        writer.write_step(0, _latest({"member_a": -73.68,
                                      "member_b": -82.86}, [0.1]), 1.0)
        row = writer.write_step(1, _latest({"member_a": -74.68,
                                            "member_b": -83.86}, [0.1]), 0.9)
        writer.close()
        assert row["energy_spread_aligned_eV"] == pytest.approx(0.0, abs=1e-12)

    def test_real_disagreement_survives_alignment(self, tmp_path):
        writer = CommitteeTraceWriter(tmp_path / "c.csv",
                                      ["member_a", "member_b"], False)
        writer.write_step(0, _latest({"member_a": -10.0, "member_b": -20.0},
                                     [0.1]), 1.0)
        row = writer.write_step(1, _latest({"member_a": -11.0,
                                            "member_b": -22.0}, [0.1]), 0.9)
        writer.close()
        assert row["energy_spread_aligned_eV"] == pytest.approx(
            1.0 / np.sqrt(2.0), abs=1e-12)

    def test_rows_are_flushed_as_they_are_written(self, tmp_path):
        """A run that dies at step 300 keeps its first 300 steps."""
        path = tmp_path / "opt_committee.csv"
        writer = CommitteeTraceWriter(path, ["member_a", "member_b"], False)
        for step in range(3):
            writer.write_step(step, _latest({"member_a": -1.0,
                                             "member_b": -3.0}, [0.1]), 0.5)
            assert len(_read(path)) == step + 1     # readable before close
        writer.close()
        assert len(writer.rows) == 3

    def test_a_member_name_with_a_comma_is_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            CommitteeTraceWriter(tmp_path / "c.csv", ["a,b", "c"], False)


class TestPerAtomFile:
    def test_one_row_per_atom_with_symbol_and_sigma(self, tmp_path):
        path = tmp_path / "opt_committee_peratom.csv"
        write_peratom_sigma(path, ["Cu", "Cu", "C", "O"],
                            [0.01, 0.02, 0.9, 0.7])
        rows = _read(path)
        assert [r["symbol"] for r in rows] == ["Cu", "Cu", "C", "O"]
        assert [int(r["atom_index"]) for r in rows] == [0, 1, 2, 3]
        assert float(rows[2]["sigma_eV_per_A"]) == pytest.approx(0.9,
                                                                 abs=1e-12)

    def test_length_mismatch_is_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            write_peratom_sigma(tmp_path / "p.csv", ["Cu"], [0.1, 0.2])


class TestFlaggingRule:
    def _rows(self, sigmas):
        return [{"step": i, "sigma_max_eV_per_A": s,
                 "energy_spread_aligned_eV": 0.0} for i, s in enumerate(sigmas)]

    def test_sigma_above_the_threshold_flags_the_configuration(self):
        summary = uncertainty_summary(
            self._rows([0.30, 0.12, 0.08]),
            _latest({"member_a": -1.0, "member_b": -3.0}, [0.02, 0.08]),
            threshold=0.05, threshold_source="fmax", symbols=["Cu", "C"])
        assert summary["flagged"] is True
        assert summary["sigma_max_final_eV_per_A"] == pytest.approx(0.08,
                                                                    abs=1e-12)
        assert summary["threshold_eV_per_A"] == pytest.approx(0.05, abs=1e-12)
        assert summary["threshold_source"] == "fmax"

    def test_sigma_below_the_threshold_does_not_flag(self):
        summary = uncertainty_summary(
            self._rows([0.30, 0.02]),
            _latest({"member_a": -1.0, "member_b": -3.0}, [0.01, 0.02]),
            threshold=0.05, threshold_source="fmax")
        assert summary["flagged"] is False

    def test_the_flag_uses_the_final_geometry_not_the_peak(self):
        """A path that passed through a strained geometry but converged to a
        well-constrained minimum is not flagged."""
        summary = uncertainty_summary(
            self._rows([3.476, 0.01]),
            _latest({"member_a": -1.0, "member_b": -3.0}, [0.005, 0.01]),
            threshold=0.05, threshold_source="fmax")
        assert summary["flagged"] is False
        assert summary["sigma_max_peak_eV_per_A"] == pytest.approx(3.476,
                                                                   abs=1e-12)
        assert summary["peak_step"] == 0

    def test_exactly_at_the_threshold_does_not_flag(self):
        summary = uncertainty_summary(
            self._rows([0.05]),
            _latest({"member_a": -1.0, "member_b": -3.0}, [0.05]),
            threshold=0.05, threshold_source="explicit")
        assert summary["flagged"] is False

    def test_the_worst_atom_is_reported_with_its_symbol(self):
        summary = uncertainty_summary(
            self._rows([0.5]),
            _latest({"member_a": -1.0, "member_b": -3.0}, [0.01, 0.5]),
            threshold=0.05, threshold_source="fmax", symbols=["Cu", "C"])
        assert summary["worst_atom"] == 1
        assert summary["worst_atom_symbol"] == "C"

    def test_an_empty_trace_reports_no_flag_rather_than_crashing(self):
        """A run that died before its first optimizer step still has to
        finish its record."""
        summary = uncertainty_summary([], None, threshold=0.05,
                                      threshold_source="fmax")
        assert summary["flagged"] is False
        assert summary["sigma_max_final_eV_per_A"] is None


class TestTraceCarriesBothSigmas:
    def test_the_row_carries_the_masked_and_unmasked_maxima(self, tmp_path):
        writer = CommitteeTraceWriter(tmp_path / "t.csv", ["a", "b"], False)
        latest = {
            "energies": {"a": -1.0, "b": -3.0},
            "energy_mean": -2.0, "sigma_max": 0.1, "sigma_mean": 0.05,
            "worst_atom": 1, "sigma_max_all": 0.9, "n_free_atoms": 1,
        }
        row = writer.write_step(0, latest, fmax_value=0.04)
        writer.close()
        assert row["sigma_max_eV_per_A"] == pytest.approx(0.1)
        assert row["sigma_max_all_eV_per_A"] == pytest.approx(0.9)
        assert row["n_free_atoms"] == 1

    def test_the_header_names_the_new_columns(self, tmp_path):
        writer = CommitteeTraceWriter(tmp_path / "t.csv", ["a", "b"], False)
        writer.close()
        header = (tmp_path / "t.csv").read_text().splitlines()[0]
        assert "sigma_max_all_eV_per_A" in header
        assert "n_free_atoms" in header


class TestPerAtomFileMarksConstrainedAtoms:
    def test_every_atom_still_gets_a_row(self, tmp_path):
        """Seeing the frozen atoms is how a reader checks the masking on
        their own run, so constrained atoms are listed, not dropped."""
        path = tmp_path / "p.csv"
        write_peratom_sigma(path, ["Cu", "C"], [0.9, 0.1],
                            sigma_free=[0.0, 0.1],
                            free_mask=[[False, False, False],
                                       [True, True, True]])
        lines = path.read_text().splitlines()
        assert len(lines) == 3
        assert lines[1].split(",")[:2] == ["0", "Cu"]

    def test_the_free_component_count_is_recorded(self, tmp_path):
        path = tmp_path / "p.csv"
        write_peratom_sigma(path, ["Cu", "C"], [0.9, 0.1],
                            sigma_free=[0.0, 0.1],
                            free_mask=[[False, False, False],
                                       [True, True, True]])
        rows = _read(path)
        assert rows[0]["free_components"] == "0"
        assert rows[1]["free_components"] == "3"
        assert float(rows[0]["sigma_free_eV_per_A"]) == pytest.approx(0.0)

    def test_a_partly_fixed_atom_counts_its_free_directions(self, tmp_path):
        path = tmp_path / "p.csv"
        write_peratom_sigma(path, ["Cu"], [0.5], sigma_free=[0.3],
                            free_mask=[[True, True, False]])
        rows = _read(path)
        assert rows[0]["free_components"] == "2"

    def test_omitting_the_mask_treats_every_atom_as_free(self, tmp_path):
        """Keeps the Python API callable with three arguments."""
        path = tmp_path / "p.csv"
        write_peratom_sigma(path, ["Cu", "C"], [0.9, 0.1])
        rows = _read(path)
        assert rows[0]["free_components"] == "3"
        assert float(rows[0]["sigma_free_eV_per_A"]) == pytest.approx(0.9)
