#!/usr/bin/env python


class InputError(Exception):
    """Raised when there is an error in the input (a file that cannot be read, energies that do not fit the conformers)."""

    pass


class ArgumentError(Exception):
    """Raised when a command-line argument or option value is invalid, or the options contradict each other."""

    pass


class EnergiesRequiredError(InputError):
    """Raised when the requested metric or option needs an energy for every conformer and some are missing."""

    pass


class UniqueError(Exception):
    """Raised when the dissimilarity matrix has only one unique entry."""

    pass


# Exit codes of the command line: 0 ok, 2 bad arguments, 3 input parse or unit problem, 4 energies required but
# missing, 5 degenerate clustering, 1 anything unexpected.
EXIT_OK = 0
EXIT_UNEXPECTED = 1
EXIT_ARGUMENTS = 2
EXIT_INPUT = 3
EXIT_ENERGIES = 4
EXIT_DEGENERATE = 5


def exit_code_of(exc):
    """The exit code of the command line for an exception, and the short name of its class."""
    if isinstance(exc, ArgumentError):
        return EXIT_ARGUMENTS, "arguments"
    if isinstance(exc, EnergiesRequiredError):
        return EXIT_ENERGIES, "energies"
    if isinstance(exc, UniqueError):
        return EXIT_DEGENERATE, "degenerate-clustering"
    if isinstance(exc, InputError):
        return EXIT_INPUT, "input"
    return EXIT_UNEXPECTED, "unexpected"
