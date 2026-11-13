#!/usr/bin/env python

"""Everything marc writes goes into one output directory.

A run first writes into a staging directory inside --outdir. Only when the whole run succeeded are the files moved into
--outdir (after the files of an earlier run have been removed, if --overwrite was given), so a run that fails leaves no
output behind and never half-replaces the results of an earlier run.
"""

import json
import os
import re
import shutil
import tempfile

from navicat_marc.exceptions import ArgumentError

SCHEMA_VERSION = "1"

# The names of the files marc writes: --overwrite replaces these and nothing else.
OWNED_FILES = re.compile(
    r"sel_\d+_in\d+\.xyz|selection\.json|[A-Za-z_]+_dendrogram\.png|tsne_plot(_legend)?\.png|(dm|edm|fm)\.npy"
)


class OutputStage:
    """The output directory of a run and the staging directory inside it."""

    def __init__(self, outdir, overwrite=False):
        self.outdir = os.path.abspath(outdir)
        self.overwrite = overwrite
        self.created = False
        self.tmp = None
        if os.path.exists(self.outdir) and not os.path.isdir(self.outdir):
            raise ArgumentError(f"--outdir {outdir} exists and is not a directory.")
        if os.path.isdir(self.outdir) and os.listdir(self.outdir) and not overwrite:
            raise ArgumentError(
                f"--outdir {outdir} is not empty. Give an empty or new directory, or --overwrite to replace the files of an earlier run."
            )
        try:
            if not os.path.isdir(self.outdir):
                os.makedirs(self.outdir)
                self.created = True
            self.tmp = tempfile.mkdtemp(prefix=".marc_stage_", dir=self.outdir)
        except OSError as e:
            raise ArgumentError(f"--outdir {outdir} cannot be created or written: {e}")

    def path(self, name=""):
        """A path inside the staging directory (the directory itself for no name)."""
        return os.path.join(self.tmp, name) if name else self.tmp

    def write_json(self, name, obj):
        with open(self.path(name), "w") as fh:
            json.dump(obj, fh, indent=1)
            fh.write("\n")

    def stale_files(self):
        """The files of an earlier run that commit() will remove (none without --overwrite)."""
        if not self.overwrite:
            return []
        return [
            name
            for name in sorted(os.listdir(self.outdir))
            if OWNED_FILES.fullmatch(name) and os.path.isfile(os.path.join(self.outdir, name))
        ]

    def commit(self):
        """Removes the files of an earlier run (with --overwrite), moves the staged files into the output directory and
        returns the names of the files removed."""
        removed = self.stale_files()
        for name in removed:
            os.remove(os.path.join(self.outdir, name))
        for name in sorted(os.listdir(self.tmp)):
            os.replace(os.path.join(self.tmp, name), os.path.join(self.outdir, name))
        shutil.rmtree(self.tmp, ignore_errors=True)
        self.tmp = None
        return removed

    def discard(self):
        """Removes the staging directory, and the output directory too if this run created it and nothing else is in it."""
        if self.tmp is not None:
            shutil.rmtree(self.tmp, ignore_errors=True)
            self.tmp = None
        if self.created:
            try:
                os.rmdir(self.outdir)
            except OSError:
                pass
