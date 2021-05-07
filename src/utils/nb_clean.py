"""
    This script provides utilities to clean noteboks in a very customized way
    as well as
    functionalities to automate it in your git procedure.

    After successfully run configure the git cleaning procedure notebooks were
    cleaned
    before they are pushed to the respository based on the cleaning method.

    This code based on the code from https://github.com/srstevenson/nb-clean

    @author jhtuhmacher
"""

import argparse
import pathlib
import subprocess
import sys

import nbformat

ATTRIBUTE = "*.ipynb filter=nb_clean"


def set_git_config():
    """
        Automatically set the right git configuration for the automated
        cleaning.

        Even if we don't have any parameters for this function the argument
        'args' is
        necessary in the function definition!

        Parameter
        ---------
            args: Commandline arguments
    """

    print("Set up git config.")

    process = subprocess.run(
        ["git"] + list([
            "config", "filter.nb_clean.clean",
            "python src/utils/nb_clean.py clean"
        ]),
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        check=False,
    )

    stdout: str = process.stdout.decode().strip()

    attributes = pathlib.Path("../../.git/info/attributes")

    if attributes.is_file() and ATTRIBUTE in attributes.read_text():
        return

    with attributes.open("a") as file:
        file.write(f"\n{ATTRIBUTE}\n")

    print("Stdout:", stdout)
    print("Git config setted.")


def clean(args: argparse.Namespace):
    """
        The function that is executed to clean a notebook.

        Parameter
        ---------
            args: Commandline arguments
    """

    nb_in = args.input
    nb_out = args.output

    notebook = nbformat.read(nb_in, as_version=nbformat.NO_CONVERT)

    for cell in notebook.cells:
        cell["metadata"] = {}
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            # cell["outputs"] = []

    nbformat.write(notebook, nb_out)


def main():
    """
        Entry point for the script.
    """
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    subparsers = parser.add_subparsers(dest="subcommand")
    subparsers.required = True

    configure_parser = subparsers.add_parser(
        "config-git",
        help="Integrate notebook cleaning in git.",
    )
    configure_parser.set_defaults(func=set_git_config)

    clean_parser = subparsers.add_parser(
        "clean",
        help="Cleans notebooks",
    )
    clean_parser.add_argument(
        "-i",
        "--input",
        type=argparse.FileType("r"),
        default=sys.stdin,
        help="input file",
    )
    clean_parser.add_argument(
        "-o",
        "--output",
        type=argparse.FileType("w"),
        default=sys.stdout,
        help="output file",
    )
    clean_parser.set_defaults(func=clean)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
