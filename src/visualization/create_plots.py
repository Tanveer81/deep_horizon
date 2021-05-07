""" Simple script to create all plots that are in the paper (21.02.2021).

    @author: jhuthmacher
"""
# pylint: disable=wrong-import-position,wrong-import-order,ungrouped-imports,import-outside-toplevel
###########
# Imports #
###########
# import sys
# sys.path.insert(0, "../")

from datetime import datetime

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

from config import log


############################
# Stlying for presentation #
############################
matplotlib.use('pdf')
plt.style.use('seaborn')
sns.set_context("paper")

def create_plots(model_path: str = "../data/model/"):
    """ Create the predefined plots.

        Parameters:
            model_path: str
                Path to the model data. Please make sure that the underlying structure
                corresponds to the expected structure.
                For details have a look at create_plots.ipynb.
    """
    ################
    # Data Loading #
    ################
    from utils.data_utils import load_data  # noqa: E402

    dataLog10, data, _, _ = load_data()

    ####################
    # Load Model Data  #
    ####################
    from utils.data_utils import load_model_data  # noqa: E402

    test_path = model_path + "obs_vs_pred_csv/test/"
    train_path = model_path + "obs_vs_pred_csv/train/"

    log.info("Load model data.")
    dfs1 = load_model_data(test_path, channels=[1, 2, 3, 4, 5])
    dfs1_train = load_model_data(train_path, channels=[1, 2, 3, 4, 5])

    ############################
    # Load Feature Importances #
    ############################
    from utils.data_utils import load_feature_importances  # noqa: E402

    fi_path = model_path + "feature_imp_csv/test/"

    log.info("Load feature importance.")
    df_feature_imp2 = load_feature_importances(fi_path, mode="test", channels=[1, 2, 3, 4, 5])

    ######################
    # Feature Importance #
    ######################
    from visualization.plot_results import generate_feature_imp_plot

    # Merge combined features
    df_feature_imp2["feature"] = df_feature_imp2["feature"].str.replace("_combined", "")

    log.info("Create feature importance plot.")
    generate_feature_imp_plot(df_feature_imp2, pivot="feature", val="perm_imp", fmt="1.3f")

    ######################
    # Correlation Matrix #
    ######################
    from visualization.plot_results import generate_correlation_matrix  # noqa: E402

    log.info("Create correlation matrix plot.")
    generate_correlation_matrix(dataLog10[dataLog10.columns[:-7]].dropna())

    ################################
    # Positional Distribution Plot #
    ################################
    from visualization.plot_results import generate_pos_distr  # noqa: E402

    log.info("Create spatial measurement counts distribution plot.")
    generate_pos_distr(data)

    #################################
    # Positional Intensity Heatmaps #
    #################################
    from visualization.plot_results import generate_pos_intensity_heatmaps  # noqa: E402

    log.info("Create spatial measurement distribution plot.")
    generate_pos_intensity_heatmaps(dataLog10)

    #######################
    # Prediction Heatmaps #
    #######################
    from visualization.plot_results import generate_pred_heatmap
    import matplotlib
    params = {'legend.fontsize': 'xx-large',
            'axes.labelsize': 'xx-large',
            'axes.titlesize': 'xx-large',
            'xtick.labelsize': 'x-large',
            'ytick.labelsize': 'x-large'}
    matplotlib.rcParams.update(params)

    log.info("Create prediction heatmaps plot.")
    channels = [0, 1, 3, 3, 4]    
    for ch in channels:
        train = dfs1_train[ch]
        test = dfs1[ch]
        generate_pred_heatmap(train, test, annotated_text=f"Channel {ch + 1}",
                              path=f".figures/prediction_heatmap_ch{ch+1}.pdf")

    #############
    # Time Plot #
    #############
    from visualization.plot_results import plot_pred_obs_time  # noqa: E402

    log.info("Create time line plots.")

    start_date = datetime(2015, 9, 19, 1, 15)
    end_date = datetime(2015, 9, 19, 20, 30)

    _ = plot_pred_obs_time(dfs1, save_plot=True, idx_range=(start_date, end_date))

    ###########################
    # Proton Feature Relation #
    ###########################
    from visualization.plot_results import generate_proton_relation_plot  # noqa: E402

    log.info("Create proton relation plots.")
    generate_proton_relation_plot(dataLog10, channel="p1")

    ######################
    # Distribution Plots #
    ######################
    from visualization.plot_results import generate_feature_distr  # noqa: E402

    log.info("Create feature distribution plots.")
    generate_feature_distr(dataLog10)
