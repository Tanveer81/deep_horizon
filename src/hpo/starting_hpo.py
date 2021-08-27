from utils.ml_flow_utils import init_ml_flow
from models.baseline_models.historical_binning_fact import HistBinFactory
from models.baseline_models.knn_fact import KNeighborsFactory

from models.treebased_models.ada_boost_fact import AdaBoostFactory
from models.treebased_models.dec_tree_fact import DecTreeFactory
from models.treebased_models.extra_trees_fact import ExtraTreesFactory
from models.treebased_models.gbr_fact import GBRFactory
from models.treebased_models.hist_gbr_fact import HistGBRFactory
from models.treebased_models.lgbm_fact import LightGBMFactory
from models.treebased_models.rand_for_fact import RandForFactory

from models.linear_models.bayesian_ridge_fact import BayesianRidgeFactory
from models.linear_models.lars_cv_fact import LarsCVFactory
from models.linear_models.lars_fact import LarsFactory
from models.linear_models.lasso_fact import LassoFactory
from models.linear_models.lasso_lars_fact import LassoLarsFactory
from models.linear_models.linear_svr_fact import LinearSVRFactory
from models.linear_models.ridge_cv_fact import RidgeCVFactory
from models.linear_models.ridge_fact import RidgeFactory

from models.mlp.mlp_fact import MLPFactory

from hpo.hpo_general_setup import hpo_all_channels


def get_model_factories() -> list:
    """
    Inits all model_factories for model-types that should be trained.
    
    :param:
        none
    :return:
        list of model factories to pass to the train()-method
    """
    model_factories = []
    
    # Treebased models
    model_factories.append(AdaBoostFactory())
    model_factories.append(DecTreeFactory())
    model_factories.append(ExtraTreesFactory())
    model_factories.append(GBRFactory())
    model_factories.append(HistGBRFactory())
    model_factories.append(LightGBMFactory())
    model_factories.append(RandForFactory())
    
    # Linear models
    model_factories.append(BayesianRidgeFactory())
    model_factories.append(LarsCVFactory())
    model_factories.append(LarsFactory())
    model_factories.append(LassoFactory())
    model_factories.append(LassoLarsFactory())
    model_factories.append(LinearSVRFactory())
    model_factories.append(RidgeCVFactory())
    model_factories.append(RidgeFactory())
    
    # Neural Nets
    model_factories.append(MLPFactory())

    # Baseline models
    model_factories.append(HistBinFactory())
    model_factories.append(KNeighborsFactory())

    return model_factories


def hpo(model_factories, num_samples: int, exp_name: str) -> None:
    """
    Traines all models provided by the input-factories
    
    :param:
        model_factories: Factories that produce model types for training
    
    :return:
        None
        
    """    
    for model_fact in model_factories:
        hpo_all_channels(model_fact=model_fact, num_samples=num_samples, exp_name=exp_name)

    return

def start_hpo(exp_name: str = 'Predicting soft proton intensities', num_samples: int = 1) -> None:
    """
    Executes HPO and logs results to mlflow

    :param: exp_name:
        Name of experiment in mlflow to log the results to
        
    :param num_samples:
        number of hpo trials per model type
    
    return:
    """    
    # Execute training
    hpo(model_factories=get_model_factories(), num_samples=num_samples, exp_name=exp_name)