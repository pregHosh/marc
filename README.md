navicat-marc: modular Analysis of Representative Conformers
==============================================
<!-- zenodo badge will go here -->

![marc logo](./images/marc_logo.png)
[![PyPI version](https://badge.fury.io/py/navicat-marc.svg)](https://badge.fury.io/py/navicat_marc)
[![DOI](https://zenodo.org/badge/544877792.svg)](https://zenodo.org/doi/10.5281/zenodo.12569950)

## Contents
* [About](#about-)
* [Install](#install-)
* [Concept](#concept-)
* [Examples](#examples-)
* [Citation](#citation-)

## About [↑](#about)

The code runs on pure python with the following dependencies: 
- `numpy`
- `scipy`
- `matplotlib`
- `scikit-learn`
- `networkx`


## Install [↑](#install)

You can install marc using pip:

```python
pip install navicat_marc
```

Afterwards, you can call marc as:

```python 
python -m navicat_marc [-h] [-version] -i INPUT [INPUT ...] --outdir OUTDIR [--overwrite] [--keep-frame] [-c C] [-m M] [-n N] [-ewin EWIN] [-sf SF] [-mine] [-yesh] [-s] [-nosymm] [-as] [-o OUTPUT_FILENAME] [--energy-source {title,efile,none}] [--energy-unit {Eh,kcal,kJ}] [-efile EFILE] [--ecol ECOL] [-v VERB] [-pm PLOTMODE] [--seed SEED]
```
or simply

```python 
navicat_marc [-h] [-version] -i INPUT [INPUT ...] --outdir OUTDIR [--overwrite] [--keep-frame] [-c C] [-m M] [-n N] [-ewin EWIN] [-sf SF] [-mine] [-yesh] [-s] [-nosymm] [-as] [-o OUTPUT_FILENAME] [--energy-source {title,efile,none}] [--energy-unit {Eh,kcal,kJ}] [-efile EFILE] [--ecol ECOL] [-v VERB] [-pm PLOTMODE] [--seed SEED]
```

Alternatively, you can download the package and execute:

```python 
python setup.py install
```

or

```python 
pip install .
```

Options can be consulted using the `-h` flag in either case. The help menu is quite detailed. 

Note that the main functions are all exposed and called directly in sequential order from `marc.py`, in case you want to incorporate them in your own code.

## Concept [↑](#concept)

Several strategies are available for the generation of conformational ensembles. Typically, one then needs to sort the ensemble and proceed with the study of the most energetically favored conformers, which will be the most accesible thermodynamically following a Boltzmann distribution.

However, sorting conformers accurately requires high quality energy computations. Accurately determining the energy of every structure may be too computationally demanding. Furthermore, upon refinement many conformers may collapse to the same stationary point. Even worse, for some tasks, knowledge of conformer interconvertibility is highly important.

To untangle these issues, marc provides a convenient way of accomplishing three goals:

- Select a handful of distinct conformers that are representative of the diversity of the conformational ensemble using combined metrics.
- Apply energy cutoffs based on the available energies to remove entire clusters from the space using the `-ewin` flag and inputting a treshold in kcal/mol.
- Proceed iteratively, helping the user select non-redundant conformers than can then be refined with a higher level and fed back to marc.

The default clustering metric used in marc is the `"avg"` distance, which measures pairwise similarity based on the average of three normalized distance matrices: heavy-atom rmsd, energy difference and kernel of the heavy-atom dihedral angles of the system. 

The logic behind this choice is that rmsd ought to be good except in cases where trivial single bond rotations increase the rmsd without affecting the energy, while the dihedral metric smooths systems that only differ by a few torsions. The possible metrics (to be fed to the `-m` flag) are `"rmsd"`, `"erel"` (based on the available energies), `"da"` (based on the most relevant dihedral angle of the molecule), `"ewrmsd"` (combining geometry and energy) and `"mix"` (combining geometry, dihedrals and energy with the DISTATIS algorithm), other than the default `avg`.

## Examples [↑](#examples)

The examples subdirectory contains some examples obtained by running [CREST](https://xtb-docs.readthedocs.io/en/latest/crest.html). Any of the xyz files can be run as:

```python
navicat_marc -i [FILENAME] --outdir [OUTPUT DIRECTORY]
```

Options can be consulted with the `-h` flag. `--outdir` is required: everything marc writes goes there.

The input of marc is either a series or xyz files or a single trajectory-like xyz file with many conformers. All structures are expected to be analogous in terms of sorting and molecular topology. An atom line is an element symbol or an atomic number followed by x, y and z (integers are fine); columns after the coordinates are ignored, and blank lines at the end of a trajectory are ignored. A structure with fewer atom lines than announced is refused.

### Energies

Energies per conformer can come from the title line of every xyz block or file (`--energy-source title`, the default), from an energy file (`--energy-source efile` with `-efile FILE`; the titles are then ignored), or not at all (`--energy-source none`; the titles are ignored). Every energy is converted to kcal/mol, and the format each one was read with is recorded in `selection.json`. Either every conformer has an energy or none has: energies for only some of the conformers are refused, and so are titles in more than one format (they may be in different units or at a different level of theory). Give the energies with `-efile` in that case.

Title formats that are recognised, and their unit:

| Title | Unit |
|---|---|
| `-345.12345678` (a bare number with a decimal point) | `--energy-unit`, default Eh |
| `E = -345.12345678` | `--energy-unit`, default Eh |
| `Energy: -345.1234` (also after a name, as `conf1.log<TAB>Energy: -345.1234`) | `--energy-unit`, default kcal/mol |
| `-345.12345678  !CONF3` (CREST) | Eh |
| `energy: -345.123 gnorm: 0.0004 ...` (xtb) | Eh |
| `Coordinates from ORCA-job name E -345.123` (ORCA) | Eh |
| `SCF Done:  E(RB3LYP) =  -345.123 ...` or `scf done: -345.123` (Gaussian) | Eh |
| `name.xyz Eopt -345.123` | Eh |
| `-345.12345678 Eh`, `-216.5 kcal/mol`, `-905.8 kJ/mol` | as written |
| `... energy=-9391.3 ...` (extended xyz, as written by ASE) | eV |

Anything else gives no energy: marc does not guess. That includes a bare integer (it is more likely a conformer number), a name, and a number followed by a count. A title without an energy is fine unless a metric or option needs energies (see below).

`--energy-unit {Eh,kcal,kJ}` is the unit of the numbers that carry none themselves (the table says which). Without it, bare numbers are in Eh and `Energy: x` is in kcal/mol, as before; the formats with a unit of their own keep it whatever `--energy-unit` says.

The energy file `-efile` has one energy per line, or an index and an energy per line (separated by blanks or a comma; the index counts the conformers in the order they were given, from 1 as CREST does or from 0, and a file whose indices do not follow that order is refused), or is a table written by [GoodVibes](https://github.com/patonlab/GoodVibes). For a GoodVibes table give the column with `--ecol`, either its name in the header (for example `E_SPC` or `qh-G(T)_SPC`) or its number (1 is the first column after the structure name). Its rows are matched to the conformers by structure name, or by position if none of the names matches. The unit is `--energy-unit`, by default Eh. The file must hold exactly the conformers given: the table of species `examples/tropanes_example/energies.dat` cannot be used for the conformers of one species, but the table of conformers `examples/tropanes_example/tsa_conformers/energies.dat` can.

```python
navicat_marc -i examples/RhCp_conformers.xyz --outdir rhcp_out -n 4 -ewin 5 --energy-source efile -efile crest.energies --energy-unit Eh
```

### Selecting

Note that, by default, marc will select the most representative conformer out of every cluster. If you can provide energy values that you trust strongly, the `mine` flag will ensure that the lowest energy conformer of every cluster is selected.

The energy window `-ewin` (kcal/mol) is measured from the global minimum, which is the lowest energy of all conformers. A cluster is kept if its lowest energy member is within `ewin` kcal/mol of the global minimum, whatever the energy of the representative marc picked for it; otherwise the entire cluster is discarded. The output states the window, the global minimum and how many clusters were kept.

If `-n` is not given, marc chooses the number of representatives itself. It clusters the ensemble for every candidate number from 1 (1 % of the conformers when there are 200 or more) up to just under half of the number of conformers, scores every clustering with the silhouette score (a single cluster scores 0), and takes the number with the best score. The choice never exceeds the number of distinct conformers according to the chosen metric.

### Output

Everything marc writes goes into the directory given with `--outdir`, which is created if it does not exist:

- `sel_<k>_in<idx0>.xyz`: the representative of every accepted cluster. `k` is the number of the cluster (`id` in `selection.json`), and `idx0` is the index of the representative in the input: the position of the file in the `-i` list, or the structure number in a trajectory, counting from 0. The title line of the input is kept as it was. The atoms are centred and rotated to their principal axes, as they are compared; with `--keep-frame` the atom lines of the input are written unchanged. Representatives of clusters rejected by `-ewin` are not written; they are listed in `selection.json`.
- `selection.json`: what was selected and why. Its keys are `schema_version`, `marc_version`, `argv`, `inputs` (for every conformer: `path`, `frame_index0`, `name`, `energy_kcal`, `energy_source`), `energy_source`, `energy_unit_in`, `ecol`, `metric`, `matrices_used`, `clustering`, `n_requested`, `n_auto` (when `-n` is not given: `method`, the `scores` of every candidate number of clusters and `k_chosen`), `seed`, `ewin_kcal`, `ewin_rule`, `mine`, `global_min_index0`, `clusters` (for every cluster: `id`, `members_index0`, `representative_index0`, `rep_rel_kcal` and `mean_rel_kcal` relative to the global minimum, `min_member_index0`, `status` accepted or rejected, `reason` for a rejected one, `output_file`) and `warnings`. Energies and the keys that depend on them are `null` when there are none.
- The plots (`-pm 1` and `-pm 2`), the matrices of `-v 4` and more (`dm.npy`, `edm.npy`) and `fm.npy` (`-v 6`), and the log of `-o NAME` are written there too, and nothing is written anywhere else.

A directory that is not empty is refused unless `--overwrite` is given. With `--overwrite` the files of an earlier run (`sel_*.xyz`, `selection.json`, plots and `.npy` files) are replaced and nothing else is removed. A run that fails leaves nothing behind and does not touch an earlier result. The `-w` flag is kept for old command lines and has no effect: the selected conformers are always written.

`navicat_marc.marc.run_marc_from_args` returns `(indices, clusters, rejected)`: the representatives, their clusters, and one boolean per representative that is `True` for a cluster rejected by `-ewin`.

### Exit codes

| Code | Meaning |
|---|---|
| 0 | done |
| 2 | bad arguments (a value that is not valid, options that contradict each other, a missing `--outdir`, a directory that is not empty without `--overwrite`) |
| 3 | an input could not be read or does not fit: an xyz file, the energy file, a number of energies that is not the number of conformers, titles in different formats, an index that is out of order |
| 4 | energies are required and missing, or are given for only some conformers |
| 5 | the clustering is degenerate (for example every structure is the same, or agglomerative clustering with one cluster) |
| 1 | anything unexpected |

On failure one line of JSON is written to standard error, for example `{"error": "energies", "exit_code": 4, "message": "Energies required: ..."}`, and no traceback.

The metrics `avg` (the default), `erel`, `ewrmsd`, `ewda` and `mix`, and the options `ewin` and `mine`, need an energy for every conformer. If energies are missing, marc stops with the message `Energies required`, exit code 4 and no output files; it does not switch to another metric. Ask for `--energy-source none` or leave the titles without energies to run a metric that does not need them. The metrics `rmsd` and `da` work without energies. The `-ewin` and `-mine` options never change the matrix of the chosen metric.

`--seed N` fixes every random step so that a run can be repeated exactly: the embedding used by kmeans (a fixed seed of 42 is used without the option), the embedding behind the automatic choice of `n` with agglomerative clustering, and affinity propagation. Without `--seed`, agglomerative clustering with automatic `n` and affinity propagation can differ from run to run.

If the atoms of a conformer file are listed in a different order than in the previous one, marc reorders them to match and says so in the output.

High verbosity levels (`-v 1`, `-v 2`, etc.) will print significantly more information while marc runs. To be as automated as possible, reasonable default values are set for most choices, but extreme verbosity can be obtained by raising the value.

marc is able to use molecular symmetry and deal with shuffling, including the effect of bond rotations and symmetries, when computing heavy atom rmsd. However, this comes at a cost, and therefore this function is deactivated automatically for large systems. You can enfoce sorting by including the flags `-s` and `-as`, where `s` uses an approximate sorting routine and `as` does brute force rmsd comparisons over isomorphisms. `nosymm` deactivates this functionality, which can lead to wrong rmsds! Luckily, you can always resort to metrics that do not require rmsd.

As a final note, marc does not consider hydrogen atoms for geometry analysis. You can force marc to include them by using the `-yesh` flag, which is generally not recommended. Obviously, including hydrogens makes any sorting much more time consuming.

## Citation [↑](#citation)

Please cite our work with the repository DOI and the manuscript that introduced the code. You can find it [here](https://doi.org/10.1021/acs.jpclett.4c01657) and in the reference:

```
Laplaza, R.; Wodrich, M. D.; Corminboeuf, C. Overcoming the Pitfalls of Computing Reaction Selectivity from Ensembles of Transition States. J. Phys. Chem. Lett. 2024, 15 (29), 7363–7370. https://doi.org/10.1021/acs.jpclett.4c01657.
```

---


