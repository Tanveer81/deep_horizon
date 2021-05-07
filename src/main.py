"""
Main script of the Deep Horizon package, which defines several entry points
(for command line and for running as script)
as well as trigger the execution of further code.

    Here we can define processes that will be executed like:
        1. Load data
        2. Preprocess data
        3. Train on model X
"""
# pylint: disable=line-too-long
import argparse
import os
import warnings

from config import log
from visualization import create_plots
from hpo.starting_hpo import start_hpo

warnings.filterwarnings('ignore')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='deep_horizon',
                                     description='Framework for predicting proton intensity.')
    parser.add_argument('--train', dest='train',
                        help='Execute the training procedure (with default settings) for the defined model. Example: --train mlp')
    parser.add_argument('--hpo', dest='hpo',
                        help='Execute hyper parameter optimization for the defined model. Example: --hpo mlp')
    parser.add_argument('--tracking', dest='tracking', default="http://localhost:5000",
                        help='Determines the trakcing URL for the MLFlow server.')
    parser.add_argument('--create_plots', dest='create_plots', default="../data/model/",
                        help=('Create predefined plots.' +
                              'Please provide a path to the model data or make sure that the data is located' +
                              'in the default location (../data/model/). Hint: The default output loaction is ".figures/".'))
    parser.add_argument('-l', dest='logging', action='store_true', help='Flag to disable logging')

    args = parser.parse_args()

    if args.logging:
        log.getLogger().setLevel(log.ERROR)

    log.info("Deep Horizon")
    os.environ["MLFLOW_TRACKING_URI"] = args.tracking
    log.info('Set tracking environment variable to %s', os.environ["MLFLOW_TRACKING_URI"])

    if args.hpo is not None:
        start_hpo()
        pass
    elif args.create_plots is not None:
        create_plots(args.create_plots)
