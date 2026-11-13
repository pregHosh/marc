#!/usr/bin/env python

"""Energies of the conformers: title-line formats, energy files and units.

Every energy is stored in kcal/mol. Where an energy comes from is recorded per conformer. A title that is not one of
the formats listed in TITLE_FORMATS gives no energy; a number is never guessed from a title that does not say what it
is.
"""

import math
import re

from navicat_marc.exceptions import ArgumentError, EnergiesRequiredError, InputError

HA_TO_KCAL = 627.509474
HA_TO_EV = 27.211386246
EV_TO_KCAL = HA_TO_KCAL / HA_TO_EV
KJ_TO_KCAL = 1.0 / 4.184

UNIT_TO_KCAL = {"Eh": HA_TO_KCAL, "kcal": 1.0, "kJ": KJ_TO_KCAL, "eV": EV_TO_KCAL}
VALID_UNITS = ["Eh", "kcal", "kJ"]
VALID_SOURCES = ["title", "efile", "none"]

# A number with a decimal point (or an exponent), and any number.
_DEC = r"[-+]?(?:\d+\.\d*|\.\d+|\d+[eE][-+]?\d+)(?:[eE][-+]?\d+)?"
_NUM = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"

# (tag, pattern, unit, default unit). The unit is fixed when the format has one of its own; otherwise it is the value
# of --energy-unit, and the default unit applies when that option is not given. The first pattern that matches wins.
TITLE_FORMATS = [
    ("scf-done", re.compile(r"scf done:\s*(?:E\([^)]*\)\s*=\s*)?(" + _DEC + ")", re.I), "Eh", None),
    ("orca", re.compile(r"\bORCA-job\b.*\bE\s+(" + _DEC + r")\s*$"), "Eh", None),
    ("xtb", re.compile(r"(?<![\w])energy:\s*(" + _NUM + ")"), "Eh", None),
    ("eopt", re.compile(r"\bEopt\s+(" + _DEC + ")"), "Eh", None),
    ("extxyz", re.compile(r'(?<![\w])energy\s*=\s*"?(' + _NUM + ")"), "eV", None),
    ("crest-conf", re.compile(r"^(" + _DEC + r")\s+!CONF\d*$"), "Eh", None),
    ("eh-suffix", re.compile(r"^(" + _DEC + r")\s*Eh$"), "Eh", None),
    ("kcal-suffix", re.compile(r"^(" + _DEC + r")\s*kcal/mol$"), "kcal", None),
    ("kJ-suffix", re.compile(r"^(" + _DEC + r")\s*kJ/mol$"), "kJ", None),
    ("e-equals", re.compile(r"^E\s*=\s*(" + _NUM + r")$"), None, "Eh"),
    ("energy-colon", re.compile(r"(?<![\w])Energy:\s*(" + _NUM + ")"), None, "kcal"),
    ("bare", re.compile(r"^(" + _DEC + r")$"), None, "Eh"),
]


def title_format_names():
    """The recognised title formats, for the help and the documentation."""
    return [t[0] for t in TITLE_FORMATS]


def parse_title_energy(title, unit=None):
    """Energy in kcal/mol and the format it was read with, from the title line of an xyz block: (None, None) if the
    title holds no energy in a recognised format.

    unit: the unit of the numbers that carry none (Eh, kcal or kJ). Without it a bare number or `E = x` is in Eh, as
    ever, and `Energy: x` is in kcal/mol, as ever. Formats with a unit of their own (xtb, ORCA, `SCF Done:`, `Eopt`,
    CREST `!CONF`, `x Eh`, extxyz `energy=` in eV) ignore it.
    """
    t = (title or "").strip()
    for tag, rx, fixed, default in TITLE_FORMATS:
        m = rx.search(t)
        if m is None:
            continue
        try:
            value = float(m.group(1))
        except ValueError:
            continue
        if not math.isfinite(value):
            continue
        u = fixed or unit or default
        return value * UNIT_TO_KCAL[u], f"title:{tag}({u})"
    return None, None


def _is_goodvibes_table(rows):
    return any(r.split()[0] == "Structure" for r in rows)


def _clean_rows(lines):
    rows = []
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#") or set(s) <= set("* -"):
            continue
        rows.append(s)
    return rows


def _goodvibes_energies(path, rows, unit, ecol, names, warnings):
    header = next(r for r in rows if r.split()[0] == "Structure")
    cols = header.split()[1:]
    if ecol is None:
        raise ArgumentError(f"The energy file {path} is a GoodVibes table: choose its energy column with --ecol (a column name or its number, 1 is the first of {cols}).")
    if ecol in cols:
        ci = cols.index(ecol)
    elif ecol.isdigit() and 1 <= int(ecol) <= len(cols):
        ci = int(ecol) - 1
    else:
        raise ArgumentError(f"--ecol {ecol!r} is not a column of the GoodVibes table {path}. Its columns are {cols}.")
    table, flagged = {}, 0
    for r in rows:
        t = r.split()
        if t[0] not in ("o", "x") or len(t) < 3:
            continue
        if t[1] in table:
            raise InputError(f"The structure {t[1]} is listed twice in the GoodVibes table {path}.")
        try:
            table[t[1]] = float(t[2 + ci])
        except (ValueError, IndexError):
            raise InputError(f"Could not read the column {cols[ci]} of the structure {t[1]} in the GoodVibes table {path}.")
        flagged += t[0] == "x"
    if not table:
        raise InputError(f"The GoodVibes table {path} has no structure rows.")
    n = len(names)
    present = [nm in table for nm in names]
    if all(present):
        values = [table[nm] for nm in names]
        how = "by name"
    elif not any(present) and len(table) == n:
        values = list(table.values())
        how = "by position"
        warnings.append(f"The structure names of the GoodVibes table {path} are not the conformer names, so its {n} rows were matched to the conformers by position.")
    else:
        raise InputError(
            f"The GoodVibes table {path} has {len(table)} structures and {n} conformers were given, and the names do not match "
            f"({sum(present)} of the {n} conformer names are in the table). A table of species cannot give the energies of conformers."
        )
    if flagged:
        warnings.append(f"{flagged} structures of the GoodVibes table {path} are marked x (imaginary frequencies).")
    return [v * UNIT_TO_KCAL[unit] for v in values], f"efile:goodvibes[{cols[ci]}]({unit}, {how})"


def read_energy_file(path, unit, ecol, names, warnings):
    """Energies in kcal/mol, one per conformer, and the format they were read with (the same for all).

    The file holds one energy per line, or `index energy` per line (blank or comma separated; the index runs 1..n or
    0..n-1 in the order of the conformers), or a GoodVibes table whose energy column is chosen with ecol.
    """
    unit = unit or "Eh"
    try:
        with open(path, "r") as fh:
            lines = fh.read().splitlines()
    except (OSError, UnicodeDecodeError) as e:
        raise InputError(f"The energy file {path} could not be found or read: {e}")
    rows = _clean_rows(lines)
    if not rows:
        raise InputError(f"The energy file {path} is empty.")
    n = len(names)
    if _is_goodvibes_table(rows):
        values, tag = _goodvibes_energies(path, rows, unit, ecol, names, warnings)
        return values, tag
    if ecol is not None:
        raise ArgumentError("--ecol is only for a GoodVibes table; this energy file has one energy or an index and an energy per line.")
    vals, idxs, extra = [], [], False
    for r in rows:
        t = r.replace(",", " ").split()
        try:
            if len(t) == 1:
                vals.append(float(t[0]))
            else:
                idxs.append(int(t[0]))
                vals.append(float(t[1]))
                extra = extra or len(t) > 2
        except ValueError:
            raise InputError(f"The line {r!r} of the energy file {path} is not an energy or an index followed by an energy.")
        if not math.isfinite(vals[-1]):
            raise InputError(f"The line {r!r} of the energy file {path} holds an energy that is not finite.")
    if idxs and len(idxs) != len(vals):
        raise InputError(f"The energy file {path} mixes lines with an index and lines without one.")
    if len(vals) != n:
        raise InputError(f"The energy file {path} holds {len(vals)} energies but {n} conformers were given.")
    if idxs:
        if idxs[0] not in (0, 1) or idxs != list(range(idxs[0], idxs[0] + n)):
            raise InputError(
                f"The indices of the energy file {path} do not follow the order of the conformers: they should run 1..{n} (or 0..{n - 1}), "
                f"they are {idxs[:6]}{' ...' if n > 6 else ''}."
            )
        if extra:
            warnings.append(f"The energy file {path} has more than two columns; only the second one was used.")
    tag = "efile:index" if idxs else "efile:one-column"
    return [v * UNIT_TO_KCAL[unit] for v in vals], f"{tag}({unit})"


def assign_energies(molecules, source, unit, efile, ecol, names, warnings):
    """Sets molecule.energy (kcal/mol, or None) and molecule.energy_source on every molecule.

    source: title (the title line of each conformer), efile (the energy file) or none (no energies).
    Either every conformer has an energy or none has: energies for only some of them are refused, and so are
    titles in more than one format.
    """
    n = len(molecules)
    if source == "none":
        values, tags = [None] * n, [None] * n
    elif source == "efile":
        values, tag = read_energy_file(efile, unit, ecol, names, warnings)
        tags = [tag] * n
    else:
        parsed = [parse_title_energy(getattr(m, "title", ""), unit) for m in molecules]
        values, tags = [p[0] for p in parsed], [p[1] for p in parsed]
        have = [v is not None for v in values]
        if any(have) and not all(have):
            first = have.index(False)
            raise EnergiesRequiredError(
                f"Energies required for every conformer or for none: only {sum(have)} of {n} titles hold an energy in a recognised format "
                f"(the first without one is conformer {first}, title {getattr(molecules[first], 'title', '')!r}). "
                "Fix the titles, give all the energies with --energy-source efile -efile FILE, or ignore them with --energy-source none."
            )
        formats = sorted({t for t in tags if t is not None})
        if len(formats) > 1:
            raise InputError(
                f"The energies in the titles come in {len(formats)} different formats ({', '.join(formats)}), so they may not be in the same unit "
                "or at the same level of theory. Use inputs with one format, or give the energies with --energy-source efile -efile FILE."
            )
    for m, v, t in zip(molecules, values, tags):
        m.energy = v
        m.energy_source = t
    return values, tags
