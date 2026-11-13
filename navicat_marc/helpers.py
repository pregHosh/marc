#!/usr/bin/env python

import argparse
import os
import re
import sys
from itertools import cycle

import networkx as nx
import numpy as np

from navicat_marc.energies import VALID_SOURCES, VALID_UNITS, assign_energies
from navicat_marc.exceptions import ArgumentError, InputError
from navicat_marc.molecule import Molecule, at_eq, b_eq
from navicat_marc.outputs import OutputStage
from navicat_marc.rmsd import kabsch_rmsd, reorder_distance, reorder_hungarian


def long_substr(data):
    """Longest common substring for basename elucidation."""
    substr = ""
    if len(data) > 1 and len(data[0]) > 0:
        for i in range(len(data[0])):
            for j in range(len(data[0]) - i + 1):
                if j > len(substr) and all(data[0][i : i + j] in x for x in data):
                    substr = data[0][i : i + j]
    while (substr[-1] == "-") or (substr[-1] == "_"):
        substr = substr[:-1]
    while (substr[0] == "-") or (substr[0] == "_"):
        substr = substr[1:]
    if substr == "":
        substr = "Molecule"
    return substr


valid_c = ["kmeans", "agglomerative", "affprop"]
valid_m = ["rmsd", "erel", "da", "ewrmsd", "ewda", "mix", "avg"]


def yesno(question):
    """Simple Yes/No Function."""
    prompt = f"{question} ? (y/n): "
    ans = input(prompt).strip().lower()
    if ans not in ["y", "n"]:
        print(f"{ans} is invalid, please try again...")
        return yesno(question)
    if ans == "y":
        return True
    return False


def bround(x, base: float = 10, type=None) -> float:
    if type == "max":
        return base * np.ceil(x / base)
    elif type == "min":
        return base * np.floor(x / base)
    else:
        tick = base * np.round(x / base)
        return tick


def chunker(seq, size):
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


def group_data_points(bc, ec, names):
    try:
        groups = np.array([str(i)[bc:ec].upper() for i in names], dtype=object)
    except Exception as m:
        raise InputError(
            f"Grouping by name characters did not work. Error message was:\n {m}"
        )
    type_tags = np.unique(groups)
    cycol = cycle("bgrcmky")
    cymar = cycle("^ospXDvH")
    cdict = dict(zip(type_tags, cycol))
    mdict = dict(zip(type_tags, cymar))
    cb = np.array([cdict[i] for i in groups])
    ms = np.array([mdict[i] for i in groups])
    return cb, ms


def molecules_from_file(filename, scale_factor=1.10, noh=True):
    """Molecules of every structure of a trajectory-like xyz file. Blank lines at the end of the file are ignored."""
    molecules = []
    try:
        with open(filename, "r") as f:
            lines = list(f.readlines())
    except (OSError, UnicodeDecodeError) as e:
        raise InputError(f"The xyz file {filename} could not be found or read: {e}")
    while lines and not lines[-1].strip():
        lines.pop()
    try:
        n_atoms = int(lines[0].strip())
    except (ValueError, IndexError):
        raise InputError(
            f"Could not obtain the number of atoms in the .xyz file {filename} from first line. Check format."
        )
    if n_atoms < 1:
        raise InputError(f"The number of atoms in the .xyz file {filename} is {n_atoms}. Check format.")

    if len(lines) % (n_atoms + 2) != 0:
        raise InputError(
            f"Could not parse trajectory xyz file {filename} properly: {len(lines)} lines do not fill whole structures of {n_atoms} atoms (a structure takes {n_atoms + 2} lines). Check format."
        )
    for k, chunk in enumerate(chunker(lines, n_atoms + 2)):
        try:
            molecule = Molecule(lines=chunk, scale_factor=scale_factor, noh=noh)
        except InputError as e:
            raise InputError(f"Structure {k} of {filename}: {e}")
        molecules.append(molecule)
    return molecules


VERSION = "0.2.8"


class _ArgumentParser(argparse.ArgumentParser):
    """An argument parser that raises ArgumentError instead of printing the usage and exiting."""

    def error(self, message):
        raise ArgumentError(message)


def build_parser():
    mbuilder = _ArgumentParser(
        prog="navicat_marc",
        description="Analyse conformer ensembles to find the most representative structures.",
        epilog="Exit codes: 0 ok, 2 bad arguments, 3 input could not be read or does not fit, 4 energies required but missing, 5 degenerate clustering, 1 unexpected. On failure one JSON line with the error is written to standard error.\nRemember to cite the marc paper or repository - \n if they have a DOI by now\n - and enjoy!",
    )
    mbuilder.add_argument(
        "-version", "--version", action="version", version=f"%(prog)s {VERSION}"
    )
    mbuilder.add_argument(
        "-i",
        "--i",
        "-input",
        dest="input",
        nargs="+",
        action="append",
        type=str,
        required=True,
        help="Filename(s) containing the conformational ensemble as an xyz trajectory, or separately as xyz files.",
    )
    mbuilder.add_argument(
        "--outdir",
        dest="outdir",
        required=True,
        help="Directory for everything marc writes: the selected conformers as sel_<k>_in<idx0>.xyz, selection.json, and the plots and .npy files. It is created if it does not exist. A directory that is not empty is refused unless --overwrite is given.",
    )
    mbuilder.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        default=False,
        help="Allow --outdir to hold files already. The files of an earlier run (sel_*.xyz, selection.json, plots and .npy files) are replaced; nothing else is removed. (default: False)",
    )
    mbuilder.add_argument(
        "--keep-frame",
        dest="keep_frame",
        action="store_true",
        default=False,
        help="Write the selected conformers with the atom lines of the input unchanged. By default they are written centred and rotated to their principal axes. (default: False)",
    )
    mbuilder.add_argument(
        "-c",
        "--c",
        "-cluster",
        "--cluster",
        dest="c",
        type=str,
        default="kmeans",
        help=f"Clustering algorithms to use. (default: kmeans)\nPossible values are: {valid_c}",
    )
    mbuilder.add_argument(
        "-m",
        "--m",
        "-metric",
        "--metric",
        dest="m",
        type=str,
        default="avg",
        help=f"Metric to use to define distance. (default: avg)\nPossible values are: {valid_m}",
    )
    mbuilder.add_argument(
        "-n",
        "--n",
        "--n_clusters",
        dest="n",
        default=None,
        help="Number of representative conformers to select. (default: chosen automatically as the number of clusters with the best silhouette score. The candidates run from 1 to just under half of the number of conformers, so at most about half of them are selected, and never more than the number of distinct conformers.)",
    )
    mbuilder.add_argument(
        "-ewin",
        "--ewin",
        dest="ewin",
        default=None,
        help="If set to a float, energy window in kcal/mol: a cluster is kept if its lowest energy member is within the window of the global minimum (the lowest energy of all conformers), otherwise the cluster is rejected. (default: None)",
    )
    mbuilder.add_argument(
        "-sf",
        "--sf",
        dest="sf",
        type=float,
        default=1.10,
        help="If set to a float, scale factor used to determine connectivity from 3D coordinates using covalent radii. (default: 1.10)",
    )
    mbuilder.add_argument(
        "-mine",
        "--mine",
        dest="mine",
        action="store_true",
        default=False,
        help="If set, the minimum energy conformer per sample will be taken instead of the centermost one. Needs energies for all conformers. (default: False)",
    )
    mbuilder.add_argument(
        "-yesh",
        "--yesh",
        dest="noh",
        action="store_false",
        default=True,
        help="If set, hydrogen atoms will be considered for all RMSD and dihedral angle computations. (default: ignore hydrogens)",
    )
    mbuilder.add_argument(
        "-s",
        "--s",
        "-sort",
        "--sort",
        dest="sort",
        action="store_true",
        default=False,
        help="If set, will attempt to sort molecular geometries within isomorphisms w.r.t. the first structure. This can be time consuming. (default: False)",
    )
    mbuilder.add_argument(
        "-nosymm",
        "--nosymm",
        dest="nosymm",
        action="store_true",
        default=False,
        help="If set, will avoid sorting molecular geometries within isomorphisms unless its requested explicitly with the allsort/sort flags. (default: False)",
    )
    mbuilder.add_argument(
        "-as",
        "--as",
        "-allsort",
        "--allsort",
        "-truesort",
        "--truesort",
        dest="allsort",
        action="store_true",
        default=False,
        help="If set, will attempt to sort molecular geometries within isomorphisms in every pairwise comparison. This is extremely time consuming in large molecules. (default: False, True if molecule is small)",
    )
    mbuilder.add_argument(
        "-o",
        "-output",
        "--o",
        "--output",
        dest="output_filename",
        default=None,
        help="If set to a filename, name of a file inside --outdir that will log marc's standard output. (default: None)",
    )
    mbuilder.add_argument(
        "--energy-source",
        dest="energy_source",
        choices=VALID_SOURCES,
        default=None,
        help="Where the energies of the conformers come from: title (the title line of each xyz block or file), efile (the file given with -efile; the xyz titles are ignored) or none (no energies; titles are ignored). (default: efile if -efile is given, otherwise title) Energies for only some of the conformers are refused.",
    )
    mbuilder.add_argument(
        "--energy-unit",
        dest="energy_unit",
        choices=VALID_UNITS,
        default=None,
        help="Unit of the energies that carry none themselves: the energies of -efile, bare numbers and `E = x` in the title, and `Energy: x` in the title. (default: Eh for -efile and bare numbers, kcal for `Energy: x`.) Title formats with a unit of their own (xtb, ORCA, `SCF Done:`, `Eopt`, CREST `!CONF`, `x Eh`, extxyz `energy=` in eV) keep it. Energies are converted to kcal/mol.",
    )
    mbuilder.add_argument(
        "-efile",
        "--efile",
        dest="efile",
        default=None,
        help="File containing the energies of each conformer: one energy per line, or an index and an energy per line (the index counts the conformers from 1, or from 0), or a GoodVibes table (then give --ecol). The unit is --energy-unit (default: Eh). (default: None)",
    )
    mbuilder.add_argument(
        "--ecol",
        dest="ecol",
        default=None,
        help="The energy column of a GoodVibes table given with -efile: its name in the header (for example E_SPC or qh-G(T)_SPC) or its number (1 is the first column after the structure name). Rows are matched to the conformers by structure name, or by position if no name matches. (default: None)",
    )
    mbuilder.add_argument(
        "-v",
        "--v",
        "--verb",
        dest="verb",
        type=int,
        default=1,
        help="Verbosity level of the code. Higher is more verbose and viceversa. Above 3 the dissimilarity matrices are saved to --outdir as .npy files. (default: 1)",
    )
    mbuilder.add_argument(
        "-pm",
        "--pm",
        "-plotmode",
        "--plotmode",
        dest="plotmode",
        type=int,
        default=0,
        help="Plotting mode. Set to 1 to generate agglomerative dendrograms and to 2 to also generate TSNE plots. The plots are saved to --outdir. (default: 0)",
    )
    mbuilder.add_argument(
        "--seed",
        dest="seed",
        type=int,
        default=None,
        help="Non-negative integer seed for every random step (the embedding used by kmeans, the automatic choice of n, affinity propagation), so that a run can be repeated exactly. (default: None; the kmeans embedding then uses a fixed seed of 42 and the other random steps are not seeded)",
    )
    mbuilder.add_argument(
        "-w",
        "--w",
        "-write",
        "--write",
        dest="write",
        action="store_false",
        default=True,
        help="Kept for old command lines and without effect: the selected conformers are always written to --outdir.",
    )
    return mbuilder


def processargs(arguments, state=None):
    """Reads the command line and the inputs.

    Returns (basename, molecules, dof, c, m, n, ewin, mine, sort, truesort, plotmode, verb, is_write, seed, run), where run
    is a dict with what the command line itself needs and run_marc_from_args does not: the argument list, the outputs
    (outdir, stage, overwrite, keep_frame, log file name), the energy source and unit, the warnings so far, and where each
    conformer came from. state, if a dict is given, receives the output stage as soon as it exists, so that the caller
    can remove it if anything fails later.
    """
    arguments = list(arguments)
    input_str = " ".join(["navicat_marc"] + arguments)
    state = {} if state is None else state
    warnings = []
    args = build_parser().parse_args(arguments)

    # Values of the options
    if args.seed is not None and args.seed < 0:
        raise ArgumentError(f"seed must be a non-negative integer, but {args.seed} was provided.")

    if args.ewin is not None:
        try:
            ewin = float(args.ewin)
        except (TypeError, ValueError):
            raise ArgumentError(f"ewin was set to {args.ewin!r} which is not a number of kcal/mol.")
        if not np.isfinite(ewin) or ewin < 0:
            raise ArgumentError(f"ewin must be a non-negative number of kcal/mol, but {args.ewin!r} was provided.")
    else:
        ewin = None

    if args.n is not None:
        try:
            n = int(args.n)
        except (TypeError, ValueError):
            raise ArgumentError(f"n must be an integer or not given, but {args.n!r} was provided.")
        if n < 1:
            raise ArgumentError(f"n must be at least 1, but {n} was provided.")
    else:
        n = None

    if args.c not in valid_c:
        raise ArgumentError(f"Unknown clustering algorithm selected. Valid algorithms are:\n {valid_c}")

    if args.m not in valid_m:
        raise ArgumentError(f"Unknown metric for clustering selected. Valid metrics are:\n {valid_m}")

    source = args.energy_source
    if source is None:
        source = "efile" if args.efile is not None else "title"
    if source == "efile" and args.efile is None:
        raise ArgumentError("--energy-source efile needs the energy file: give it with -efile FILE.")
    if source != "efile" and args.efile is not None:
        raise ArgumentError(f"-efile was given but --energy-source is {source}. Use --energy-source efile to read the energies from the file.")
    if args.ecol is not None and source != "efile":
        raise ArgumentError("--ecol selects a column of the GoodVibes table given with -efile.")

    log_name = args.output_filename
    if log_name is not None:
        if re.search(r"[^A-Za-z0-9_.\-]", log_name) or log_name.strip(".") == "":
            raise ArgumentError(f"The provided output filename {log_name} is invalid: use letters, digits, _, - and . only; it is written inside --outdir.")

    # The output directory: refused here if it holds files and --overwrite is not given
    stage = OutputStage(args.outdir, args.overwrite)
    state["stage"] = stage
    if log_name is not None:
        ofile = open(stage.path(log_name), "w")
        state["log"] = (sys.stdout, ofile)
        sys.stdout = ofile
    print(f"Executing marc version {VERSION} with input arguments:\n{input_str}")

    # The inputs
    input_list = [item for sublist in args.input for item in sublist]
    if args.verb > 2:
        print(f"Input files are {input_list}.")
    try:
        sources = []  # path and frame index of every conformer
        if len(input_list) > 1:
            terminations = list(set([i[-3:] for i in input_list]))
            basename = long_substr([filename.split("/")[-1] for filename in input_list])
            if len(terminations) > 1 or terminations[0] != "xyz":
                raise InputError(
                    f"Files with {terminations} instead of all xyz termination fed as input. Exiting."
                )
            molecules = []
            for i in input_list:
                molecule = Molecule(filename=i, scale_factor=args.sf, noh=args.noh)
                if molecule.truncated:
                    raise InputError(f"The xyz file {i} is truncated: {molecule.truncated}.")
                molecules.append(molecule)
            sources = [(os.path.abspath(i), 0) for i in input_list]
        else:
            basename = input_list[0].split("/")[-1].split(".")[0]
            termination = input_list[0][-3:]
            if termination == "xyz":
                molecules = molecules_from_file(
                    input_list[0], scale_factor=args.sf, noh=args.noh
                )
                width = max(int(np.floor(np.log10(len(molecules))) + 1), 2)
                for idx, molecule in enumerate(molecules):
                    molecule.name = f"{basename}_{idx:0{width}d}"
                sources = [(os.path.abspath(input_list[0]), idx) for idx in range(len(molecules))]
            else:
                raise InputError(
                    f"File with {termination} instead of xyz termination fed as input. Exiting."
                )

        # The output directory must not hold the inputs: --overwrite would replace them
        inside = sorted({p for p, _ in sources if os.path.commonpath([os.path.realpath(os.path.dirname(p)), os.path.realpath(stage.outdir)]) == os.path.realpath(stage.outdir)})
        if inside:
            raise ArgumentError(f"The input file {inside[0]} is inside --outdir. Give an --outdir that does not hold the inputs.")

        # Check for number of molecules
        if len(molecules) < 3:
            raise InputError("Less than three molecules provided. Exiting.")
        if n is not None and n > len(molecules):
            raise ArgumentError(f"n is {n} but only {len(molecules)} conformers were given.")

        # Energies: from the titles or the energy file; all or none
        names = [molecule.name for molecule in molecules]
        assign_energies(molecules, source, args.energy_unit, args.efile, args.ecol, names, warnings)
        if source == "efile" and args.verb > 0:
            print(f"Energies of the {len(molecules)} conformers read from {args.efile} ({args.energy_unit or 'Eh'}) and converted to kcal/mol. Double check the ordering!")
        if source == "efile" and args.verb > 1:
            for i, molecule in enumerate(molecules):
                print(f"Molecule {i} set with energy {molecule.energy} kcal/mol")

        # Check for atom ordering and number
        dof, natoms = get_dof(molecules, verb = args.verb, warnings=warnings)

        sort = False

        if args.verb > 0 and sort:
            print(
                "Warning! The given molecule geometries were not sorted. Molecules have been reordered to try to fix this.\n This does not guarantee that the RMSDs will be properly calculated! If your input structures are completely shuffled, all calculated RMSD-based metrics might be bad."
            )

        # Check for isomorphism
        for molecule_a, molecule_b in zip(molecules, molecules[1:]):
            g_a = molecule_a.graph
            g_b = molecule_b.graph

            GM = nx.algorithms.isomorphism.GraphMatcher(g_a, g_b, at_eq, b_eq)
            if GM.is_isomorphic():
                im = True
            else:
                if args.verb > 0:
                    print("Warning! Molecule topologies are not isomorphic.")
                im = False
                break
    except (InputError, ArgumentError):
        raise
    except (ValueError, KeyError, IndexError, ArithmeticError, np.linalg.LinAlgError) as e:
        raise InputError(f"The input could not be read: {type(e).__name__}: {e}")

    # Double check if energies are properly set
    energies = [molecule.energy for molecule in molecules]
    if None in energies:
        if args.verb > 2:
            print(f"Energies are: {energies}")

    if args.sort != sort:
        sort = args.sort
    if args.allsort:
        if args.verb > 0:
            print(
                "Warning! All pairs isomorphism-based sorting is remarkably expensive. This option is available for cases in which the input structures are isomorphically shuffled.\n This may take a ridiculous amount of time, as all isomorphisms must be checked, so you may want to explore other options or use a metric that does not require RMSDs."
            )
        truesort = True
        sort = True
    elif natoms < 50 and not args.nosymm:
        if args.verb > 0:
            print(
                "Activating symmetry-aware isomorphic sorting because the system is small. This can be avoided using the nosymm flag."
            )
        truesort = True
        sort = True
    else:
        truesort = False
    if args.m in ["erel", "da", "ewda"]:
        sort = False
        truesort = False
        if args.verb > 0:
            print("Warning! Will not sort molecules because no RMSDs will be computed.")
    if ((sort and truesort) or sort) and not im:
        raise InputError(
            f"Input molecules cannot be reliably sorted.\n The automatic sorting routine will fail because the reconstructed molecular graphs were found not to be isomorphic. It might be possible to fix this by adapting the covalent radius scale factor of the graph constructor with the -sf option, but sometimes it cannot be fixed (e.g. different conformers have significantly different bond lengths). For those cases, you can use a metric that does not require RMSDs.\n Exiting."
        )

    run = {
        "argv": ["navicat_marc"] + arguments,
        "stage": stage,
        "overwrite": args.overwrite,
        "keep_frame": args.keep_frame,
        "sources": sources,
        "energy_source": source,
        "energy_unit_in": args.energy_unit if args.energy_unit is not None else ("Eh" if source == "efile" else None),
        "ecol": args.ecol,
        "warnings": warnings,
    }

    return (
        basename,
        np.array(molecules, dtype=object),
        dof,
        args.c,
        args.m,
        n,
        ewin,
        args.mine,
        sort,
        truesort,
        args.plotmode,
        args.verb,
        args.write,
        args.seed,
        run,
    )


def test_molecules_from_file(path="navicat_marc/test_files/"):
    filenames = [
        "3_h2o_conformers.xyz",
    ]
    for filename in filenames:
        molecules = molecules_from_file(f"{path}{filename}", noh=False)
        for molecule in molecules:
            print(molecule.energy, molecule.coordinates, molecule.atoms)
            assert molecule.energy is not None


# ------------------------------------------------------------------------------------------------------------------- # 

def get_dof(molecules, verb = 1, warnings = None):
    """
        Get the degrees of freedom for a list of molecules.
        Based on code obtained from the older marc version.

        Args:
        -----
            molecules: List of Molecule objects.

            verb: Integer
                Verbosity level of the code.

            warnings: (Optional) List
                Receives the text of the warning when atoms had to be reordered.

        Returns:
        -------
            dof: Integer
    """

        # Check for atom ordering and number
    sort = False
    reordered = []
    for i, (molecule_a, molecule_b) in enumerate(zip(molecules, molecules[1:]), start=1):
        atoms_a = molecule_a.atoms_with_h
        atoms_b = molecule_b.atoms_with_h
        if len(atoms_a) == len(atoms_b):
            if not all(atoms_a == atoms_b):
                reordered.append(i)
                if verb > 4:
                    print(
                        "Attempting reordering and updating molecular geometries because atom lists were not identical."
                    )
                dview = reorder_distance(
                    atoms_a,
                    atoms_b,
                    molecule_a.coordinates_with_h,
                    molecule_b.coordinates_with_h,
                )
                dres, _ = kabsch_rmsd(
                    molecule_a.coordinates_with_h, molecule_b.coordinates_with_h[dview]
                )
                if verb > 4:
                    print(f"After distance reordering, RMSD is {dres}.")
                hview = reorder_hungarian(
                    atoms_a,
                    atoms_b[dview],
                    molecule_a.coordinates_with_h,
                    molecule_b.coordinates_with_h[dview],
                )
                hres, _ = kabsch_rmsd(
                    molecule_a.coordinates_with_h,
                    molecule_b.coordinates_with_h[dview][hview],
                )
                if verb > 4:
                    print(f"After hungarian reordering on top of it, RMSD is {hres}.")
                if hres < dres:
                    molecule_b.update_with_h(
                        atoms_b[dview][hview],
                        molecule_b.coordinates_with_h[dview][hview],
                    )
                else:
                    molecule_b.update_with_h(
                        atoms_b[dview], molecule_b.coordinates_with_h[dview]
                    )
        else:
            raise InputError("Molecules do not have the same number of atoms. Exiting.")
        natoms = len(atoms_b)
        if natoms == 1:
            raise InputError("Molecules are monoatomic. Exiting.")
        elif natoms == 2:
            dof = 1
        else:
            dof = 3 * len(atoms_b) - 6
        if not dof > 0:
            raise InputError("Molecules have less than 1 degree of freedom. Exiting.")

    if reordered:
        message = (
            f"The atoms of {len(reordered)} of {len(molecules)} molecules were listed in a different order than in the molecule before them. "
            f"Their atoms were reordered to match it (molecule indices {reordered[:10]}{' ...' if len(reordered) > 10 else ''}). "
            "This does not guarantee that the RMSDs will be properly calculated."
        )
        if warnings is not None:
            warnings.append(message)
        if verb > 0:
            print("Warning! " + message)

    return dof, natoms

# ------------------------------------------------------------------------------------------------------------------- # 

def get_basename(list_files):
    """
        Get the basename for a list of xyz files used to create the Molecule objects.

        Args:
        -----
            list_files: List of filenames.

        Returns:
        -------
            basename: String
    """
    basename = long_substr([filename.split("/")[-1] for filename in list_files])

    return basename

# ------------------------------------------------------------------------------------------------------------------- # 
