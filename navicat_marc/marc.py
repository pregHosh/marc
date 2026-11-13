#!/usr/bin/env python

import json
import os
import sys
import traceback

import numpy as np
import scipy

from navicat_marc.clustering import (
    affprop_clustering,
    agglomerative_clustering,
    kmeans_clustering,
    plot_dendrogram,
    plot_tsne,
    unique_nm,
)
from navicat_marc.da import da_matrix
from navicat_marc.distatis import run_distatis
from navicat_marc.erel import erel_matrix
from navicat_marc.exceptions import (
    EnergiesRequiredError,
    InputError,
    UniqueError,
    exit_code_of,
)
from navicat_marc.helpers import VERSION, processargs
from navicat_marc.outputs import SCHEMA_VERSION
from navicat_marc.molecule import Molecule
from navicat_marc.rmsd import rmsd_matrix

from navicat_marc import helpers

# The distance matrices that make up each metric
MATRICES_USED = {
    "rmsd": ["rmsd"],
    "erel": ["erel"],
    "da": ["da"],
    "ewrmsd": ["erel", "rmsd"],
    "ewda": ["erel", "da"],
    "mix": ["rmsd", "erel", "da"],
    "avg": ["rmsd", "erel", "da"],
}


def _restore_stdout(state):
    log = state.pop("log", None)
    if log is not None:
        sys.stdout = log[0]
        log[1].close()


def _error_line(exc):
    """The one line written to standard error when a run fails: a JSON object, never a traceback."""
    code, kind = exit_code_of(exc)
    where = ""
    if code == 1:
        frames = traceback.extract_tb(exc.__traceback__)
        if frames:
            where = f" at {os.path.basename(frames[-1].filename)}:{frames[-1].lineno}"
    message = str(exc).strip() or type(exc).__name__
    if code == 1:
        message = f"Unexpected {type(exc).__name__}: {message}{where}"
    return code, json.dumps({"error": kind, "exit_code": code, "message": message})


def run_marc():
    """The command line. Returns the exit code: 0 ok, 2 bad arguments, 3 the input could not be read or does not fit,
    4 energies required but missing, 5 degenerate clustering, 1 unexpected. A failure writes one JSON line to standard
    error and leaves no output behind."""
    state = {}
    committed = False
    try:
        try:
            (
                basename,
                molecules,
                dof,
                c,
                m,
                n_clusters,
                ewin,
                mine,
                sort,
                truesort,
                plotmode,
                verb,
                is_write,
                seed,
                run,
            ) = processargs(sys.argv[1:], state)
            stage = run["stage"]
            report = {"warnings": list(run["warnings"])}

            indices, clusters, rejected = run_marc_from_args(basename = basename, 
                                                   molecules = molecules, 
                                                   dof = dof, 
                                                   c = c, 
                                                   m = m, 
                                                   n_clusters = n_clusters,
                                                   ewin = ewin,
                                                   mine = mine,
                                                   sort = sort,
                                                   truesort = truesort,
                                                   plotmode = plotmode,
                                                   verb = verb,
                                                   is_write = is_write,
                                                   seed = seed,
                                                   outdir = stage.path(),
                                                   keep_frame = run["keep_frame"],
                                                   report = report,
                                                   )

            stale = stage.stale_files()
            if stale:
                report["warnings"].append(
                    f"--overwrite: {len(stale)} files of an earlier run were removed from the output directory: {', '.join(stale[:10])}{' ...' if len(stale) > 10 else ''}."
                )
            stage.write_json(
                "selection.json",
                _selection(run, molecules, report, c, m, n_clusters, ewin, mine, seed),
            )
            _restore_stdout(state)
            stage.commit()
            committed = True
        finally:
            _restore_stdout(state)
            if not committed and "stage" in state:
                state["stage"].discard()
    except SystemExit:
        raise
    except Exception as e:
        code, line = _error_line(e)
        print(line, file=sys.stderr)
        return code

    n_written = sum(1 for cl in report["clusters"] if cl["output_file"])
    n_rejected = sum(1 for cl in report["clusters"] if cl["status"] == "rejected")
    print(
        f"marc wrote {n_written} selected conformer{'s' if n_written != 1 else ''} and selection.json to {stage.outdir}"
        + (f" ({n_rejected} rejected cluster{'s' if n_rejected != 1 else ''} listed in selection.json only)" if n_rejected else "")
    )
    return 0


def _selection(run, molecules, report, c, m, n_clusters, ewin, mine, seed):
    """The content of selection.json."""
    inputs = []
    for molecule, (path, frame) in zip(molecules, run["sources"]):
        inputs.append(
            {
                "path": path,
                "frame_index0": int(frame),
                "name": molecule.name,
                "energy_kcal": None if molecule.energy is None else float(molecule.energy),
                "energy_source": getattr(molecule, "energy_source", None),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "marc_version": VERSION,
        "argv": run["argv"],
        "inputs": inputs,
        "energy_source": run["energy_source"],
        "energy_unit_in": run["energy_unit_in"],
        "ecol": run["ecol"],
        "metric": m,
        "matrices_used": report["matrices_used"],
        "clustering": c,
        "n_requested": n_clusters,
        "n_auto": report["n_auto"],
        "seed": seed,
        "ewin_kcal": ewin,
        "ewin_rule": report["ewin_rule"],
        "mine": bool(mine),
        "global_min_index0": report["global_min_index0"],
        "clusters": report["clusters"],
        "warnings": report["warnings"],
    }


def run_marc_from_args(basename, 
                       molecules, 
                       dof: int = None, 
                       c: str = 'kmeans', 
                       m:str = 'avg', 
                       n_clusters: int = None, 
                       ewin: float = None, 
                       mine: bool = False, 
                       sort: bool = False, 
                       truesort: bool = False, 
                       plotmode: int = 0, 
                       verb: int = 1,
                       is_write: bool = False,
                       seed: int = None,
                       outdir: str = None,
                       keep_frame: bool = False,
                       report: dict = None
                       ):
    """
        Clusters conformer data into conformer ensembles. 
        Details are found in the github folder: https://github.com/lcmd-epfl/marc
        and the paper: https://doi.org/10.1021/acs.jpclett.4c01657

        Args:
        -----
            basename: 

            molecules:

            dof: (Optional) Integer
                Default: None
                Degrees of freedom.
                If not specified, calculated from the given molecules.

            c: (Optional) String
                Default: kmeans
                Clustering algorithm used.
                Options: kmeans, agglomerative, affprop

            m: (Optional) String
                Default: avg
                Metric used to define distance between clusters.
                Options: rmsd, erel, da, ewrmsd, ewda, mix, avg

            n_clusters: (Optional) Integer
                Default: None
                Number of representatitive conformers to select. If not given, the number with the best
                silhouette score is used. The candidates run from 1 (1 % of the conformers if there are
                200 or more) up to just under half of the number of conformers, and the choice never
                exceeds the number of distinct conformers.

            ewin: (Optional) Float
                Default: None
                Energy window in kcal/mol. A cluster is kept if its lowest-energy member is within the
                window of the lowest energy of all conformers (the global minimum); the other clusters
                are rejected.

            mine: (Optional) Boolean
                Default: False
                If set, the minimum energy conformer per sample will be taken instead of the centermost one.
                Requires energies for all conformers.

            sort: (Optional) Boolean
                Default: False
                If True, will atempt to sort molecular geometries within isomorphisms w.r.t. the first structure.
                Time consuming.

            truesort: (Optional) Boolean
                Default: False

            plotmode: (Optional) Integer
                Default: 0
                Set to 1 to generate agglomerative dendrograms and to 2 to also generate TSNE plots.

            verb: (Optional) Integer
                Default: 1 
                Verbosity level of the code, Higher is more verbose and vice versa.

            is_write: (Optional) Boolean
                Default: False
                Only used without outdir: write the selected conformers to the working directory as
                <name>_accepted.xyz and <name>_rejected.xyz.

            seed: (Optional) Integer
                Default: None
                Seed for every random step: the embedding used by kmeans (42 if no seed is given),
                the embedding and the gap statistic behind the automatic choice of n, and affinity
                propagation. Without a seed, only the kmeans embedding is fixed.

            outdir: (Optional) String
                Default: None
                Directory that receives everything that is written: the accepted representatives as
                sel_<k>_in<idx0>.xyz (k is the position of the cluster in the returned lists, idx0 the index of
                the representative), the plots and the .npy files. Without it the plots and .npy files go to the
                working directory and the representatives are only written if is_write is set.

            keep_frame: (Optional) Boolean
                Default: False
                Write the representatives with the atom lines of the input unchanged instead of centred and
                rotated.

            report: (Optional) Dict
                Default: None
                If given, receives what happened, for selection.json: matrices_used, n_auto, clusters (members,
                representative, relative energies, status, reason and file of every cluster), ewin_rule,
                global_min_index0 and warnings.

        Returns:
        --------
            indices: List
                Indices of the representative molecule in a cluster.
                Correlates to cluster with the same index.

            clusters: List of lists
                Clustered conformer ensembles.
                Representative molecule has same index in indices list.

            rejected: List of bool
                One entry per representative: True if its cluster was rejected by the energy window.

    """
    if dof == None: # If degrees of freedom is not specified, then calculate.
        dof, _ = helpers.get_dof(molecules, verb = verb)


    # Fill in molecule data
    l = len(molecules)
    energies = [molecule.energy for molecule in molecules]  # Retrieve the energy from every molecule in the list if applicable.

    if verb > 0:
        if n_clusters is not None:
            print(
                f"marc has detected {l} molecules in input.\n Will select {n_clusters} most representative conformers using {c} clustering and {m} as metric."
            )
        elif n_clusters is None:
            print(
                f"marc has detected {l} molecules in input.\n Will automatically select a set of representative conformers using {c} clustering and {m} as metric."
            )
        
        if not None in energies:
            print(
                "Energies for each conformer have been provided.\n Use -mine to select the lowest energy conformer of every cluster instead of the centremost one."
            )
        if ewin is not None:
            print(
                f"An energy window of {ewin} kcal/mol will be applied: a cluster is kept if its lowest energy member is within the window of the global minimum."
            )

    # ------------------------------------------------------------------------------------------ # 
    # Energies are required by the energy metrics, mine and ewin: refuse before any work is done.
    # ---------------------------------------------------------------------------------------- #

    energy_metrics = ["erel", "ewrmsd", "ewda", "mix", "avg"]
    needs_energies = []
    if m in energy_metrics:
        needs_energies.append(f"the metric {m}")
    if ewin is not None:
        needs_energies.append("-ewin")
    if mine:
        needs_energies.append("-mine")
    if needs_energies and None in energies:
        if verb > 2:
            print(f"Energies are: {energies}")
        n_missing = sum(e is None for e in energies)
        raise EnergiesRequiredError(
            f"Energies required: {' and '.join(needs_energies)} {'needs' if len(needs_energies) == 1 else 'need'} an energy for every conformer, "
            f"but {n_missing} of {l} conformers have none. Provide the energies in the title line of "
            "each conformer or with -efile, or use a metric that does not need energies (rmsd or da) "
            "without -ewin and -mine."
        )

    # ------------------------------------------------------------------------------------------ # 
    # Generate the desired metric matrix
    # ---------------------------------- #

    # ---------------------------------- #
    # RMSD matrix 
    # ---------------------------------- # 

    if m in ["rmsd", "ewrmsd", "mix", "avg"]:
        rmsd_m, rmsd_max = rmsd_matrix(molecules, sort=sort, truesort=truesort)
        if plotmode > 0:
            plot_dendrogram(rmsd_m * rmsd_max, "RMSD", verb, outdir=outdir)
        A = rmsd_m
        if verb > 4:
            print("\n The rmsd dissimilarity matrix is :\n")
            print(np.array_str(A, precision=2, suppress_small=True))
            print(
                f"Before normalization, largest dissimilarity was {rmsd_max} angstrom."
            )

    # ---------------------------------- #
    # Relative energy matrix 
    # ---------------------------------- # 

    if m in energy_metrics:
        erel_m, erel_max = erel_matrix(molecules)
        if plotmode > 0:
            plot_dendrogram(erel_m * erel_max, "E_rel", verb, outdir=outdir)
        A = erel_m
        if verb > 4:
            print("\n The relative energy dissimilarity matrix is :\n")
            print(np.array_str(A, precision=2, suppress_small=True))
            print(
                f"Before normalization, largest dissimilarity was {erel_max} kcal/mol."
            )

    # ---------------------------------- #
    # Dihedral angles 
    # ---------------------------------- # 

    if m in ["da", "ewda", "mix", "avg"]:
        da_m, da_max = da_matrix(molecules, mode="dfs")
        if plotmode > 0:
            plot_dendrogram(da_m * da_max, "Dihedral", verb, outdir=outdir)
        A = da_m
        if verb > 4:
            print("\n The dihedral angle dissimilarity matrix is :\n")
            print(np.array_str(A, precision=2, suppress_small=True))
            print(
                f"Before normalization, largest dissimilarity was {da_max} adimensional units (kernel of dihedral angle vectors)."
            )

    # ---------------------------------- #
    # Mix the metric matrices if desired
    # ---------------------------------- # 

    if m == "ewrmsd":
        A = run_distatis([erel_m, rmsd_m], verb)

    if m == "ewda":
        A = run_distatis([erel_m, da_m], verb)

    if m == "mix":
        A = run_distatis([da_m, erel_m, rmsd_m], verb)

    if m == "avg":
        A = (erel_m + da_m + rmsd_m) / 3

    if verb > 3:
        print("\n The current dissimilarity matrix is :\n")
        print(np.array_str(A, precision=2, suppress_small=True))
        print(
            f"Writing dissimilarity matrix (with full precision) to dm.npy in the output directory."
        )
        np.save(os.path.join(outdir or ".", "dm.npy"), A)

    # ------------------------------------------------------------------------------------------ # 
    # Cluster using the dissimilarity matrix of choice
    # ------------------------------------------------ # 

    auto_info = {} if n_clusters is None else None  # how the number of clusters was chosen

    if c == "kmeans":
        indices, clusters = kmeans_clustering(n_clusters, A, dof, verb, seed=seed, outdir=outdir, info=auto_info)

    if c == "agglomerative":
        indices, clusters = agglomerative_clustering(n_clusters, A, dof, verb, seed=seed, outdir=outdir, info=auto_info)

    if c == "affprop":
        indices, clusters = affprop_clustering(A, verb, seed=seed)
        if auto_info is not None:
            auto_info.update(method="affinity propagation", scores={}, k_chosen=len(indices))

    # ------------------------------------------------ # 
    # Removal of duplicates.
    # ------------------------------------------------ # 

    curr_n = len(indices)
    effA = A[indices, :][:, indices]
    if verb > 3:
        print(
            f"\n The dissimilarity matrix of the selected conformers {indices} is :\n"
        )
        print(np.array_str(effA, precision=2, suppress_small=True))
        print(
            f"Writing dissimilarity matrix of selected conformers (with full precision) to edm.npy in the output directory."
        )
        np.save(os.path.join(outdir or ".", "edm.npy"), effA)

    # The members of every cluster, as the report and the energy window see them
    member_lists = [[int(j) for j in cluster] for cluster in clusters]
    warnings = [] if report is None else report.setdefault("warnings", [])

    n, umask = unique_nm(effA, verb)
    if n < curr_n:
        print(
            f"Reduced the number of selected conformers by {curr_n - n} due to uniqueness."
        )
        if n_clusters is not None and n < n_clusters:
            raise UniqueError(
                """The desired number of clusters is larger than the number of unique entries according to the chosen similarity matrix. Exiting."""
            )
        # A cluster that is dropped as a duplicate hands its members to the kept cluster closest to it
        kept = np.where(umask)[0]
        for dropped in np.where(~umask)[0]:
            target = int(kept[np.argmin(effA[dropped, kept])])
            member_lists[target].extend(member_lists[int(dropped)])
            warnings.append(
                f"The cluster of conformer {int(indices[dropped])} was dropped because its representative is a duplicate in the chosen metric; its members were added to the cluster of conformer {int(indices[target])}."
            )
        member_lists = [member_lists[int(k)] for k in kept]
        indices = list(np.array(indices, dtype=int)[umask])
        clusters = list(np.array(clusters, dtype=object)[umask])

    # ------------------------------------------------ # 
    # Resample clusters based on energies if requested
    # ------------------------------------------------ # 

    if mine:
        if None in energies:
            if verb > 2:
                print(f"Energies are: {energies}")
            raise InputError(
                """One or more molecules do not have an associated energy. Cannot use
                 the mine option. Exiting."""
            )
        gmine = np.min(np.array(energies, dtype=float))
        for i, (index, cluster) in enumerate(zip(indices, clusters)):
            if verb > 2:
                print(
                    f"Resampling {list(cluster)} for which representative {indices[i]} has been selected."
                )
            idx_mine = cluster[
                np.argmin(np.array([energies[j] for j in cluster], dtype=float))
            ]
            if verb > 3:
                cenergies = np.around(
                    np.array([energies[j] for j in cluster]) - gmine, decimals=4
                )
                print(f"The corresponding relative energies were: {list(cenergies)}")
            if index != idx_mine:
                if verb > 2:
                    print(
                        f"Minimal energy representative {idx_mine} was selected instead."
                    )
                indices[i] = idx_mine

    # ------------------------------------------------ # 
    # If requested, discard the clusters that are too
    # high in energy: a cluster is kept if its lowest
    # energy member is within ewin of the global minimum
    # ------------------------------------------------ # 

    rejected = np.zeros((len(indices)))
    reasons = [None] * len(indices)
    if ewin is not None:
        all_energies = np.array(energies, dtype=float)
        gmin_idx = int(np.argmin(all_energies))
        gmin = all_energies[gmin_idx]
        for i, cluster in enumerate(member_lists):
            if verb > 2:
                print(
                    f"Going through {list(cluster)} for which representative {indices[i]} was selected."
                )
            lowest = np.min(all_energies[np.asarray(cluster, dtype=int)])
            gap_to_min = lowest - gmin
            if gap_to_min <= ewin:
                if verb > 1:
                    print(
                        f"Accepting selected conformer number {indices[i]}: the lowest energy member of its cluster is {gap_to_min:.2f} kcal/mol above the global minimum, within the energy window."
                    )
            else:
                if verb > 1:
                    print(
                        f"Removed selected conformer number {indices[i]}: the lowest energy member of its cluster is {gap_to_min:.2f} kcal/mol above the global minimum, outside the energy window."
                    )
                rejected[i] = 1
                reasons[i] = f"the lowest energy member of the cluster is {gap_to_min:.2f} kcal/mol above the global minimum, outside the energy window of {ewin} kcal/mol"
        if verb > 0:
            print(
                f"Energy window of {ewin} kcal/mol applied relative to the global minimum (conformer {gmin_idx}, {gmin:.4f} kcal/mol): a cluster is kept if its lowest energy member is within the window. Kept {int(len(indices) - rejected.sum())} of {len(indices)} clusters."
            )
    rejected_mask = [bool(x) for x in rejected]

    # ------------------------------------------------------------------------------------------ # 
    # Output name generation
    # ------------------------ # 

    outnames = [molecule.name for molecule in molecules]
    if None in outnames:
        format_string = f"0{max(int(np.floor(np.log10(l)) + 1), 2)}d"
        outnames = [f"{basename}_{idx:{format_string}}" for idx in range(l)]

    # ------------------------ #
    # Plot tsne
    # ------------------------ #

    if plotmode > 1:
        plot_tsne(A, indices, clusters, outnames, outdir=outdir)

    # ------------------------ #
    # Write the representative molecules that were accepted
    # ------------------------ #

    output_files = [None] * len(indices)
    if outdir is not None:
        for i, idx in enumerate(indices):
            if not rejected_mask[i]:
                output_files[i] = f"sel_{i}_in{int(idx)}.xyz"
                molecules[idx].write(os.path.join(outdir, f"sel_{i}_in{int(idx)}"), keep_frame=keep_frame)
    elif is_write:
        for i, idx in enumerate(indices):
            if rejected_mask[i]:
                molecules[idx].write(f"{outnames[idx]}_rejected")
            else:
                molecules[idx].write(f"{outnames[idx]}_accepted")

    if report is not None:
        report["matrices_used"] = MATRICES_USED[m]
        report["n_auto"] = auto_info if auto_info is None else {
            "method": auto_info.get("method", "default"),
            "scores": auto_info.get("scores", {}),
            "k_chosen": int(auto_info.get("k_chosen", len(indices))),
        }
        report["ewin_rule"] = (
            "a cluster is kept if its lowest energy member is within ewin of the global minimum (the lowest energy of all conformers)"
            if ewin is not None
            else None
        )
        have_energies = None not in energies
        rel = None
        report["global_min_index0"] = None
        if have_energies:
            e = np.array(energies, dtype=float)
            report["global_min_index0"] = int(np.argmin(e))
            rel = e - np.min(e)
        report["clusters"] = []
        for i, idx in enumerate(indices):
            members = sorted(member_lists[i])
            entry = {
                "id": i,
                "members_index0": members,
                "representative_index0": int(idx),
                "rep_rel_kcal": None if rel is None else float(rel[int(idx)]),
                "min_member_index0": None if rel is None else int(members[int(np.argmin(rel[members]))]),
                "mean_rel_kcal": None if rel is None else float(np.mean(rel[members])),
                "status": "rejected" if rejected_mask[i] else "accepted",
                "reason": reasons[i],
                "output_file": output_files[i],
            }
            report["clusters"].append(entry)
        if n_clusters is not None and c == "affprop":
            warnings.append(f"-n {n_clusters} was given but affinity propagation chooses the number of clusters itself.")
        elif n_clusters is not None and len(indices) != n_clusters:
            warnings.append(f"{len(indices)} representatives were selected for -n {n_clusters}.")

    return indices, clusters, rejected_mask
