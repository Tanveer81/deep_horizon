'''
    Permumation Importance Calculation
'''

from models.model_analysis import get_perm_importances
from data.data_loading import DataPipeline
from data.data_init import init_data

import torch
import pickle

outlier = False
test_run = False
pipe = DataPipeline(test=test_run, version=0.23, load='train+test', trans_type="robust", trans_outlier=outlier, data_outlier=outlier)


def calculate_permimps(model_file_path: str, result_file: str, channel: str):
    '''
    SUMMARY
        Calculates and logs the Permutation Importance of a model.

    PARAMETERS
        model_file_path --- str: Path of the model to calculate the permutation importances
        result_file --- str: Name of the file in which the results will be logged.
        channel --- str: The channel the model is trained on

    '''
    
    model = pickle.load(open(model_file_path, 'rb'))
    
    init_data()
    
    _, _, _, _, x_test, y_test, _, _, features = pipe.create_dataset(channel=channel)
    
    perm_imp = get_perm_importances(model, x_test, y_test)
    print("features", features)
    result_file = open(result_file, 'a')
    for i in range(0, len(perm_imp["importances_mean"])):
        line = features[i]+":"+ str(perm_imp["importances_mean"][i])+"\n"
        print(line)
        result_file.write(line)
    result_file.close()

