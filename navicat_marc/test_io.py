#!/usr/bin/env python

"""Tests of the energies, the output directory and the exit codes of the command line."""

import json
import os

import pytest

from navicat_marc.energies import (
    EV_TO_KCAL,
    HA_TO_KCAL,
    KJ_TO_KCAL,
    parse_title_energy,
    read_energy_file,
)
from navicat_marc.exceptions import ArgumentError, InputError
from navicat_marc.marc import run_marc

E = -345.12345678
EXAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")


@pytest.mark.parametrize(
    "title, unit, kcal",
    [
        (f"  {E:.8f}", None, E * HA_TO_KCAL),
        (f"  {E:.8f}", "kcal", E),
        (f"  {E:.8f}  !CONF3", None, E * HA_TO_KCAL),
        (f" energy: {E:.8f} gnorm: 0.0004 xtb: 6.7.1 (edcfbbe)", None, E * HA_TO_KCAL),
        (f"Coordinates from ORCA-job conf E {E:.8f}", None, E * HA_TO_KCAL),
        (f"SCF Done:  E(RB3LYP) =  {E:.8f}", None, E * HA_TO_KCAL),
        (f"scf done: {E:.8f}", None, E * HA_TO_KCAL),
        (f"conf0.xyz Eopt {E:.8f}", None, E * HA_TO_KCAL),
        (f'Properties=species:S:1:pos:R:3 energy={E:.8f} pbc="F F F"', "kcal", E * EV_TO_KCAL),
        (f"E = {E:.8f}", None, E * HA_TO_KCAL),
        (f"{E:.8f} Eh", "kcal", E * HA_TO_KCAL),
        ("conf0.log\tEnergy: -345.1235", None, -345.1235),
        ("conf0.log\tEnergy: -345.1235", "Eh", -345.1235 * HA_TO_KCAL),
        ("conf0.log\tEnergy: -345.1235", "kJ", -345.1235 * KJ_TO_KCAL),
    ],
)
def test_title_energy(title, unit, kcal):
    value, source = parse_title_energy(title, unit)
    assert value == pytest.approx(kcal, rel=1e-9)
    assert source.startswith("title:")


@pytest.mark.parametrize("title", ["7", "conformer_7", "", f"{E:.8f}        12", "Eopt abc"])
def test_title_without_energy(title):
    assert parse_title_energy(title) == (None, None)


def test_energy_file_forms(tmp_path):
    names = ["a", "b", "c"]
    values = [-1.5, -1.25, -1.0]
    expected = [v * HA_TO_KCAL for v in values]
    forms = {
        "one.dat": "\n".join(str(v) for v in values) + "\n",
        "one_based.dat": "\n".join(f"{i + 1} {v}" for i, v in enumerate(values)) + "\n",
        "zero_based.dat": "\n".join(f"{i},{v}" for i, v in enumerate(values)) + "\n",
    }
    for name, text in forms.items():
        path = tmp_path / name
        path.write_text(text)
        got, source = read_energy_file(str(path), None, None, names, [])
        assert got == pytest.approx(expected)
        assert source.startswith("efile:")
    path = tmp_path / "kj.dat"
    path.write_text("-100.0\n-90.0\n-80.0\n")
    got, _ = read_energy_file(str(path), "kJ", None, names, [])
    assert got == pytest.approx([-100.0 * KJ_TO_KCAL, -90.0 * KJ_TO_KCAL, -80.0 * KJ_TO_KCAL])


def test_energy_file_refused(tmp_path):
    names = ["a", "b", "c"]
    for text in ("1 -1.5\n2 -1.25\n", "3 -1.5\n2 -1.25\n1 -1.0\n", "1 -1.5\n2 x\n3 -1.0\n", "-1.5\n-1.25\n-1.0\n-0.5\n"):
        path = tmp_path / "bad.dat"
        path.write_text(text)
        with pytest.raises(InputError):
            read_energy_file(str(path), None, None, names, [])
    with pytest.raises(ArgumentError):
        (tmp_path / "gv.dat").write_text("   Structure   E_SPC   E\n o  a  -1.5  -1.4\n o  b  -1.4  -1.3\n o  c  -1.3  -1.2\n")
        read_energy_file(str(tmp_path / "gv.dat"), None, "no_such_column", names, [])


def test_goodvibes_table(tmp_path):
    path = tmp_path / "gv.dat"
    path.write_text(
        "   Structure   E_SPC   E\n   ************\n o  b  -1.4  -1.3\n o  a  -1.5  -1.4\n o  c  -1.3  -1.2\n"
    )
    for ecol in ("E_SPC", "1"):
        got, source = read_energy_file(str(path), None, ecol, ["a", "b", "c"], [])
        assert got == pytest.approx([-1.5 * HA_TO_KCAL, -1.4 * HA_TO_KCAL, -1.3 * HA_TO_KCAL])
        assert "goodvibes" in source
    got, _ = read_energy_file(str(path), None, "E", ["a", "b", "c"], [])
    assert got == pytest.approx([-1.4 * HA_TO_KCAL, -1.3 * HA_TO_KCAL, -1.2 * HA_TO_KCAL])


def run(monkeypatch, capsys, *args):
    monkeypatch.setattr("sys.argv", ["navicat_marc", *args])
    code = run_marc()
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def error_line(err):
    lines = [l for l in err.splitlines() if l.startswith("{")]
    assert len(lines) == 1 and "Traceback" not in err
    return json.loads(lines[0])


def test_exit_codes(tmp_path, monkeypatch, capsys):
    water = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_files", "3_h2o_conformers.xyz")
    out = str(tmp_path / "out")
    code, _, err = run(monkeypatch, capsys, "-i", water)  # no --outdir
    assert code == 2 and error_line(err)["exit_code"] == 2
    code, _, err = run(monkeypatch, capsys, "-i", water, "--outdir", out, "-ewin", "3kcal")
    assert code == 2 and error_line(err)["error"] == "arguments"
    code, _, err = run(monkeypatch, capsys, "-i", water, "--outdir", out, "-n", "four")
    assert code == 2
    code, _, err = run(monkeypatch, capsys, "-i", water, "--outdir", out, "--energy-source", "none")  # metric avg needs energies
    assert code == 4 and "Energies required" in error_line(err)["message"]
    code, _, err = run(monkeypatch, capsys, "-i", str(tmp_path / "missing.xyz"), "--outdir", out)
    assert code == 3
    assert not os.path.exists(out), "a failed run leaves nothing behind"


def test_selection_and_outdir(tmp_path, monkeypatch, capsys):
    rhcp = os.path.join(EXAMPLES, "RhCp_conformers.xyz")
    if not os.path.exists(rhcp):
        pytest.skip("the bundled example is not installed")
    out = str(tmp_path / "out")
    args = ["-i", rhcp, "-n", "4", "-ewin", "5", "--seed", "1", "--outdir", out]
    code, _, _ = run(monkeypatch, capsys, *args)
    assert code == 0
    with open(os.path.join(out, "selection.json")) as fh:
        selection = json.load(fh)
    assert len(selection["inputs"]) == 21 and all(i["energy_source"].startswith("title:") for i in selection["inputs"])
    accepted = [c for c in selection["clusters"] if c["status"] == "accepted"]
    assert accepted and len(accepted) < len(selection["clusters"])
    for c in accepted:
        assert c["output_file"] == f"sel_{c['id']}_in{c['representative_index0']}.xyz"
        with open(os.path.join(out, c["output_file"])) as fh:
            assert fh.read().splitlines()[0] == "73"
    assert sorted(os.listdir(out)) == sorted(["selection.json"] + [c["output_file"] for c in accepted])
    # a second run is refused, and with --overwrite replaces the first
    code, _, err = run(monkeypatch, capsys, *args)
    assert code == 2 and error_line(err)["error"] == "arguments"
    code, _, _ = run(monkeypatch, capsys, *args[:-2], "--outdir", out, "--overwrite", "-n", "2")
    assert code == 0
    with open(os.path.join(out, "selection.json")) as fh:
        assert len(json.load(fh)["clusters"]) == 2
    assert len([f for f in os.listdir(out) if f.startswith("sel_")]) <= 2
