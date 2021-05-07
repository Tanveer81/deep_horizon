""" Containing functions to generate plots.

    @author: jhuthmacher
"""
# pylint: disable=bare-except,too-many-lines,too-many-branches,too-many-statements,reimported,unbalanced-tuple-unpacking
from typing import Any

import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import cm
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

import pandas as pd
import numpy as np
import scipy as sp


# pylint: disable = line-too-long
PROTON_LABEL = r"$[\frac{log_{10} (Proton \ intensity) }{(s \cdot cm^2 \cdot steradian \cdot keV)}]$"  # noqa: E501


##################################
# High Level Generator Functions #
##################################
def generate_rel_plot(data: pd.DataFrame, path=".figures/feature_relation.pdf"):
    """ Creates relation plot.

        Parameter:
            data: pd.DataFrame
                Data that is plotted.
            path: str
                Path where the plot is stored.
    """
    for log in (False, True):
        facet = sns.relplot(
            data=data,
            col="channel",
            x="time",
            y="feature_importance",
            hue="column",
            kind="line",
            estimator="mean",
            ci="sd",
            col_wrap=3,
            err_style="bars",
            marker="o",
            facet_kws=dict(sharey=False),
        )
        for ax in facet.axes.flat:
            ax.axhline(0, color="black", ls="dashed", alpha=0.5)
            if log:
                ax.set_xscale("log")
                ax.set_xlabel(r"$\log \Delta t$ [h]")
            else:
                ax.set_xlabel(r"$\Delta t$ [h]")
        plt.savefig(f"{path.replace('.pdf', '')}{'_log' if log else ''}.pdf")


# pylint: disable = line-too-long
def generate_feature_imp_plot(data: pd.DataFrame, path: str = ".figures/feature_importance.pdf",  # noqa: E501
                              plot_type: str = "matrix", figsize=(9, 5), orientation="h",
                              pivot=None, error=None, val=None, fmt="6.0f"):
    """ Generates feature importance plot.

        Parameters:
            data: pd.DataFrame
                Contains the feature importances in matrix form, i.e. columns are the features and
                the index are the channels.
            path: str
                Path where the plot is stored as PDF. Important, the path has to contain the name
                of the file.
            plot_type: str ["matrix", "bar"]
                Determines which kind of plot is used, i.e. matrix (heatmap) or a grouped bar plot.
            figsize: tuple
                Defines the matplotlib figure size of the plot.
            orientation: str ["h", "v"]
                Defines how the grouped bar plot is organized. Either horizontally ("h") or
                vertically ("v").
            pivot: str
                The column name in data that is used to pivot the data frame. Only used for the
                grouped bar plot.
            error: str
                The column name that determines the error values. Only used for the grouped bar plot.
            val: str
                The column name that determines the acutal values that should be plotted.
                Only used for the grouped bar plot.
            fmt: str
                Format string to control the appearance of the numbers in the matrix (heatmapt) plot.
    """
    data = data.reindex(sorted(data.columns), axis=1)

    if plot_type == "matrix":
        if pivot is not None:
            data = data[[pivot, val]].pivot(columns=pivot, values=val)

        fig, _ = plot_simple_heatmap(data, cbar=True, set_aspect=True,
                                     figsize=(1.5 * 7, 1.5 * 4),
                                     xlabel="Features", ylabel="Channel",
                                     tick_freq=1, cbar_pos="top",
                                     cbar_label="", annot=True, fmt=fmt)
    elif plot_type == "bar":
        df_feature_imp = data.reindex(sorted(data.columns), axis=1)

        fig, ax = grouped_bar_plot(df_feature_imp, swap=True, space_factor=3,
                                   orientation=orientation,
                                   title="Feature Importances", value_label="",
                                   figsize=figsize, error=error, val=val,
                                   pivot=pivot)

        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    fig.tight_layout()

    fig.savefig(path, dpi=300, tight_layout=True)


def generate_correlation_matrix(data: pd.DataFrame, path: str = ".figures/correlation_matrix.pdf"):  # noqa: E501
    """ Generate the correlation matrix for the features.

        Parameters:
            data: pd.DataFrame
                Data frame containing the the whole data set, i.e. all features plus all channels.
            path: str
                Path where the plot is stored as PDF. Important, the path has to contain the
                name of the file.
    """
    # Pairwise pearson correlation
    correlation_df = data.corr()

    mask = np.triu(np.ones_like(correlation_df, dtype=np.bool))

    _, _ = plt.subplots(figsize=(11, 9))
    # ax.set_title("Correlation Matrix (Complete Data Set)")

    # annot_kws: Control for the annotation, i.e. the numbers in the sqaures
    svm = sns.heatmap(correlation_df.round(2).replace(-0, 0),
                      cmap=mpl.cm.get_cmap('RdYlBu_r'),
                      mask=mask,
                      square=True,
                      annot=True,
                      annot_kws={"size": 8},
                      #   fmt=".1g",
                      linewidths=0,
                      cbar_kws={"shrink": 1})

    svm.get_figure().savefig(path, dpi=300, transparent=True, tight_layout=True)


def generate_pos_distr(data: pd.DataFrame, path: str = ".figures/spatial_measurement_counts_distr.pdf"):
    """ Generate the positional measurement distribution plot.

        Produces a plot with three subplots, for each dimension (x,y,z) one, that portrays the
        number of measurments at some position.

        Parameters:
            data: pd.DataFrame
                Data frame containing at least the columns ["x", "y", "z", "p1"]
            path: str
                Path where the plot is stored as PDF. Important, the path has to contain the
                name of the file.
        Return:
            matplotlib.figure.Figure: Figure containing the plot.
    """
    ###############
    # Create Grid #
    ###############
    fig = plt.figure(figsize=(10, 3), constrained_layout=True)

    gs = GridSpec(1, 3, figure=fig)

    ax_xz = fig.add_subplot(gs[0])
    ax_yz = fig.add_subplot(gs[1])
    ax_xy = fig.add_subplot(gs[2])

    # inset_axes creates an additional axis within another axis.
    cbar_axis = inset_axes(ax_xy,
                           width="5%",     # width = 5% of parent_bbox width
                           height="100%",  # height : 50%
                           loc='center',
                           bbox_to_anchor=(0.7, 0., 1, 1),
                           bbox_transform=ax_xy.transAxes,
                           borderpad=0)

    ###############
    # Create bins #
    ###############
    counts_xz, max_xz, min_xz = create_bin_matrix(data, cols=["x", "z"],
                                                  resolution=1, symmetric=False)
    counts_yz, max_yz, min_yz = create_bin_matrix(data, cols=["y", "z"], resolution=1,
                                                  symmetric=False)
    counts_xy, max_xy, min_xy = create_bin_matrix(
        data, cols=["x", "y"], resolution=1, symmetric=False)

    ##################################
    # Configure coloring and colobar #
    ##################################
    cmap = mpl.cm.get_cmap('viridis')
    norm = mpl.colors.Normalize(vmin=min(min_xz, min_yz, min_xy),
                                vmax=max(max_xz, max_yz, max_xy), clip=False)
    cnorm = mpl.colors.LogNorm(vmin=1, vmax=max(max_xz, max_yz, max_xy))

    cb1 = mpl.colorbar.ColorbarBase(cbar_axis, cmap=cmap, norm=cnorm)
    cb1.set_label("Counts")

    ###############
    # Create bins #
    ###############
    ax = plot_simple_heatmap(counts_xz.replace(0, np.nan),
                             xlabel="XGSE[Re]", ylabel="ZGSE[Re]", ax=ax_xz, cmap=cmap, norm=norm)
    ax = plot_simple_heatmap(counts_yz.replace(0, np.nan),
                             xlabel="YGSE[Re]", ylabel="ZGSE[Re]", ax=ax_yz, cmap=cmap, norm=norm)
    ax = plot_simple_heatmap(counts_xy.replace(0, np.nan),
                             xlabel="XGSE[Re]", ylabel="YGSE[Re]", ax=ax_xy, cmap=cmap, norm=norm)

    fig.savefig(path, dpi=300, transparent=True, bbox_inches="tight")

    return ax.figure


def generate_pos_intensity_heatmaps(data: pd.DataFrame, path: str = ".figures/spatial_measurements_distr.pdf"):
    """ Generate the positional intensity heatmap.

        Produces for each dimension (x,y,z) a subplot that shows the average (mean) proton
        intensity for channel 1 at the specific position.

        Parameters:
            data: pd.DataFrame
                Data frame containing at least the columns ["x", "y", "z", "p1"]
            path: str
                Path where the plot is stored as PDF. Important, the path has to contain the
                name of the file.
        Return:
            matplotlib.figure.Figure: Figure containing the plot.
    """
    ###############
    # Create Grid #
    ###############

    fig = plt.figure(figsize=(10, 3), constrained_layout=True)

    gs = GridSpec(1, 3, figure=fig)

    ax_xz = fig.add_subplot(gs[0])
    ax_xz.set_facecolor("white")

    ax_yz = fig.add_subplot(gs[1])
    ax_yz.set_facecolor("white")

    ax_xy = fig.add_subplot(gs[2])
    ax_xy.set_facecolor("white")

    # inset_axes creates an additional axis within another axis.
    cbar_axis = inset_axes(ax_xy,
                           width="5%",     # width = 5% of parent_bbox width
                           height="100%",  # height : 50%
                           loc='center',
                           bbox_to_anchor=(0.7, 0., 1, 1),
                           bbox_transform=ax_xy.transAxes,
                           borderpad=0)

    ###############
    # Create bins #
    ###############
    counts_xz, max_xz, min_xz = create_bin_matrix(data, cols=["x", "z"], features=data["p1"],
                                                  resolution=1, aggr="mean", symmetric=False)
    counts_yz, max_yz, min_yz = create_bin_matrix(data, cols=["y", "z"], features=data["p1"],
                                                  resolution=1, aggr="mean", symmetric=False)
    counts_xy, max_xy, min_xy = create_bin_matrix(data, cols=["x", "y"], features=data["p1"],
                                                  resolution=1, aggr="mean", symmetric=False)

    ##################################
    # Configure coloring and colobar #
    ##################################
    cmap = mpl.cm.get_cmap('viridis')
    norm = mpl.colors.Normalize(vmin=min(min_xz, min_yz, min_xy), vmax=max(max_xz, max_yz, max_xy))
    cnorm = mpl.colors.LogNorm(vmin=0.5, vmax=10**max(max_xz, max_yz, max_xy))
    # norm=mpl.colors.LogNorm(vmin=0.5, vmax=max(max_xz, max_yz, max_xy))

    cb1 = mpl.colorbar.ColorbarBase(cbar_axis, cmap=cmap, norm=cnorm)
    cb1.set_label(PROTON_LABEL)

    ###############
    # Create bins #
    ###############
    ax = plot_simple_heatmap(
        counts_xz, xlabel="XGSE[Re]", ylabel="ZGSE[Re]", ax=ax_xz, cmap=cmap, norm=norm)
    ax = plot_simple_heatmap(
        counts_yz, xlabel="YGSE[Re]", ylabel="ZGSE[Re]", ax=ax_yz, cmap=cmap, norm=norm)
    ax = plot_simple_heatmap(
        counts_xy, xlabel="XGSE[Re]", ylabel="YGSE[Re]", ax=ax_xy, cmap=cmap, norm=norm)

    fig.savefig(path, dpi=300, transparent=True, bbox_inches="tight")

    return ax.figure


def generate_pred_heatmap(train: pd.DataFrame, test: pd.DataFrame, path: str = ".figures/prediction_heatmap.pdf",
                          annotated_text: str = "Channel X"):
    """ Generates a heatmap visualization of the predictions vs. labels.

        Parameters:
            train: pd.DataFrame
                Data frame containing the predictions and corresponding labels on the train data.
            test: pd.DataFrame
                Data frame contianing the predictions and corresponding labels on the test data.
    """
    M, bx, by = create_bin_matrix_v2(np.log10(train["Labels"]), np.log10(train["Predictions"]), 40, 40)
    M1, _, _ = create_bin_matrix_v2(np.log10(test["Labels"]), np.log10(test["Predictions"]), bx, by)
    matrices = [M, M1]

    fig, _ = plot_simple_heatmap_v2(matrices)
    fig.text(-0.05, 0.5, annotated_text, ha='center', va='center', rotation='vertical',
             fontsize="large", fontweight='bold')

    fig.savefig(path, bbox_inches='tight')


def generate_pred_heatmaps_depr(dfs_train: [pd.DataFrame], dfs: [pd.DataFrame] = None,
                                lim_left: tuple = None, lim_right: tuple = None,
                                fig_args: dict = {"figsize": (5, 5), "title": ""},
                                r_args: dict = {}, l_args: dict = {},
                                path: str = ".figures/prediction_heatmap_depr.pdf",
                                ax: matplotlib.axes.Axes = None):
    """ DEPRICATED. Use generate_pred_heatmap.
        Generates the pred. vs. obs. heatmap.

        Paramter:
            dfs_train: [pd.DataFrame]
                List of data frames from which always the FIRST ELEMENT is displayed.
                dfs_train corresponds to the left plot.
            dfs: [pd.DataFrame]
                List of data frames from which always the FIRST ELEMENT is displayed.
                dfs corresponds to the right plot. Can be None and this case only the plot shows
                only one heatmap.
            lim_left: tuple
                Tuples defining the min and max limit of the the left plot.
            lim_right: tuple
                Tuples defining the min and max limit of the the right plot.
            fig_args: dict
                Figure argumentsfor the plot.
            r_args: dict
                Arguments for the right axis.
            l_args: dict
                Arguments for the left axis.
            path: str
                Path where the plot is stored.
            ax: matplotlib.axes.Axes
                Axis which should be used instead of creating a new axis.
    """

    train = dfs_train[0]
    test = None
    ax_test = None

    if dfs is None:
        # Create Subplots
        if ax is None:
            fig, ax = plt.subplots(figsize=fig_args["figsize"])
    else:
        if ax is None:
            fig, ax = plt.subplots(1, 2, figsize=fig_args["figsize"], sharey=True)
        ax_test = ax[1]

        ax = ax[0]

        test = dfs[0]

        if lim_right is not None:
            test = test[(test["Labels"] <= 10**lim_right[0]) & (test["Predictions"] <= 10**lim_right[1])]
    fig = ax.figure

    if "title" in fig_args:
        # fig.suptitle(fig_args["title"])
        fig.text(-0.05, 0.5, fig_args["title"], ha='center', va='center',
                 rotation='vertical', fontsize="large", fontweight='bold')

    if lim_left is not None:
        train = train[(train["Labels"] <= 10**lim_left[0]) & (train["Predictions"] <= 10**lim_left[1])]

    # Train
    # pylint: disable=bad-continuation
    counts_train, max_train, min_train = create_bin_matrix(train.transform({
        "Labels": np.log10,
        "Predictions": np.log10}).replace([np.inf, -np.inf], np.nan).dropna(),
        cols=["Labels", "Predictions"],
        num_bins=30, zero_centered=False)

    # For the case we only plot a single heatmap
    if dfs is not None:
        # Train
        # pylint: disable=bad-continuation
        counts_test, max_test, min_test = create_bin_matrix(test.transform({
            "Labels": np.log10,
            "Predictions": np.log10}).replace([np.inf, -np.inf], np.nan).dropna(),
            cols=["Labels", "Predictions"],
            num_bins=30, zero_centered=False)
    else:
        min_test, max_test = min_train, max_train

    # Colorbar Configuration
    # cmap = mpl.cm.get_cmap('viridis')
    norm = mpl.colors.Normalize(vmin=min(min_test, min_train),
                                vmax=max(max_train, max_test), clip=True)
    # cnorm = mpl.colors.LogNorm(vmin=1, vmax=max(max_train, max_test), clip=True)
    # fig.subplots_adjust(right=0.95)
    # cax = fig.add_axes([0.95, 0.15, 0.02, 0.7])

    _, counts_train = plot_simple_heatmap(counts_train.replace(0, np.nan), ax=ax,
                                          distr=True, dist_scale="log",
                                          diag=True, norm=norm, cbar=False, tick_freq=4,
                                          freq_type="ind",
                                          title=f"{'Train Set: ' if dfs is not None else ''}Predictions vs. Observation",  # noqa: E501
                                          ylabel="Prediction \n" + PROTON_LABEL,
                                          #   xlabel="Observation \n" + PROTON_LABEL,
                                          set_aspect=True)

    # For the case we only plot a single heatmap
    if dfs is not None:
        _, counts_test = plot_simple_heatmap(counts_test.replace(0, np.nan), ax=ax_test,
                                             distr=True, dist_scale="log", diag=True,
                                             norm=norm, cbar=False, tick_freq=4,
                                             freq_type="ind", set_aspect=True)

    if ax_test is not None:
        ax_test.set(**r_args)

    ax.set(**l_args)
    ax.yaxis.get_label().set_fontsize(28)
    ax.xaxis.get_label().set_fontsize(28)
    ax_test.xaxis.get_label().set_fontsize(28)

    cax = fig.add_axes([1, 0.1, 0.02, 0.8])
    # cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=cnorm)
    cax.yaxis.set_ticks_position('left')
    cax.set_title("Counts")
    _ = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=mpl.cm.get_cmap('viridis')), cax=cax)

    fig.tight_layout()
    plt.tight_layout()
    fig.subplots_adjust(top=0.9, wspace=0.2)
    fig.savefig(path, dpi=300, transparent=True, tight_layout=True,
                bbox_inches="tight")


def generate_proton_relation_plot(data: pd.DataFrame, path: str = ".figures/proton_intensities_relations.pdf",
                                  channel: str = "p1"):
    """ Generates the proton intensity for channel 1 related to each feature (Figure 3).

        Parameters:
                data: pd.DataFrame
                    Data frame containing the the whole data set, i.e. all features
                    plus all channels.
                path: str
                    Path where the plot is stored as PDF. Important, the path has
                    to contain the name of the file.
                channel: str ["p1", "p2", "p3", "p4" ,"p5"]
                    Channel that is used to create the plot.
    """
    fig, ax = plt.subplots(5, 3, figsize=(16, 17))

    # a)
    plot_proton_relation(data, feature="rdist", channel=channel, xlabel="rdist", ylabel=PROTON_LABEL,  # noqa: E501
                         title="a)", linestyle="o", ax=ax[0][0], automatic_ticks=True)
    # b)
    plot_proton_relation(data, feature="z", channel=channel, xlabel="ZGSE[RE]",
                         label="ZGSE", title="b)", ax=ax[0][1], automatic_ticks=True)
    # c)
    plot_proton_relation(data, feature="y", channel=channel, xlabel="YGSE[RE]",
                         label="YGSE", title="c)", linestyle="o", ax=ax[0][2])
    # d)
    plot_proton_relation(data, feature="x", channel=channel, xlabel="XGSE[RE]", ylabel=PROTON_LABEL,
                         label="XGSE", title="d)", linestyle="o", ax=ax[1][0])
    # e)
    plot_proton_relation(data[data["VySW_GSE"].between(-300, 300)], feature="VySW_GSE",
                         channel=channel, xlabel="Solar Wind Speed, km/s",
                         label="VySW_GSE", title="e)", ax=ax[1][1])
    plot_proton_relation(data[data["VzSW_GSE"].between(-300, 300)], feature="VzSW_GSE",
                         channel=channel, xlabel="Solar Wind Speed, km/s",
                         label="VzSW_GSE", title="e)", ax=ax[1][1])
    # f)
    plot_proton_relation(data, feature="VxSW_GSE", channel=channel, xlabel="Solar Wind Speed, km/s",
                         label="VxSW_GSE", title="f)", ax=ax[1][2])
    # g)
    plot_proton_relation(data, feature="Temp", channel=channel, xlabel="Solar Wind Temperature, K",
                         ylabel=PROTON_LABEL, label="", title="g)", linestyle="o", ax=ax[2][0])
    if channel == "p1":
        ax[2][0].set_xlim(-0.5e6, 5.5e6)
        ax[2][0].set_ylim(2, 5)
    # h)
    plot_proton_relation(data, feature="NpSW", channel=channel, xlabel="Solar Wind Density, cm^-3",
                         label="", title="h)", linestyle="o", ax=ax[2][1])
    # i)
    plot_proton_relation(data, feature="Pdyn", channel=channel,
                         xlabel="Solar Wind Dynamic Pressure, nPa",
                         label="", title="i)", linestyle="o", ax=ax[2][2])
    # j)
    plot_proton_relation(data, feature="BimfxGSE", channel=channel, xlabel="IMF, nT",
                         label="BimfxGSE", title="j)", linestyle="o", ax=ax[3][0])
    plot_proton_relation(data, feature="BimfyGSE", channel=channel, xlabel="IMF, nT",
                         label="BimfyGSE", title="j)", linestyle="o", ax=ax[3][0])
    plot_proton_relation(data, feature="BimfzGSE", channel=channel, xlabel="IMF, nT",
                         ylabel=PROTON_LABEL, label="BimfzGSE", title="j)", linestyle="o",
                         ax=ax[3][0])
    # k)
    plot_proton_relation(data, feature="F107", channel=channel, xlabel="F10.7, sfu",
                         label="", title="k)", linestyle="o", ax=ax[3][1])
    # l)
    plot_proton_relation(data, feature="AE_index", channel=channel, xlabel="AE index, nT",
                         label="", title="l)", linestyle="o", ax=ax[3][2])
    # m)
    plot_proton_relation(data, feature="SYM-H_index", channel=channel, xlabel="SYM-H index, nT",
                         ylabel=PROTON_LABEL, label="", title="m)", linestyle="o", ax=ax[4][0])

    # Will be removed after we decide which features to show.
    ax[4][1].remove()
    ax[4][2].remove()

    fig.tight_layout()
    fig.savefig(path, dpi=300, transparent=False, tight_layout=True)


def generate_feature_distr(data: pd.DataFrame, path: str = ".figures/feature_distributions.pdf"):
    """ Generate the distribution plots per feature. It outputs one large matrix of distr. plots.

        Parameters:
            data: pd.DataFrame
                Data frame containing the the whole data set, i.e. all features plus all channels.
            path: str
                Path where the plot is stored as PDF. Important, the path has to contain the name
                of the file.
    """
    # Features that are plotted as subplots.
    # This has to match the dimensions of the grid, i.e. it has to match with the variables
    # rows and cols
    # Those are used as key and title.
    features = [
        ['x', 'y', 'z', 'rdist'],
        ['AE_index', 'SYM-H_index', 'F107', 'BimfxGSE'],
        ['BimfyGSE', 'BimfzGSE', 'VxSW_GSE', 'VySW_GSE'],
        ['VzSW_GSE', 'NpSW', 'Pdyn', 'Temp'],
        ['p1', 'p2', 'p3', 'p4']
    ]
    units = [
        ['XGSE[RE]', 'YGSE[RE]', 'ZGSE[RE]', 'rdist'],
        ['nT', 'nT', 'sfu', 'nT'],
        ['nT', 'nT', 'Solar Wind Speed, km/s', 'Solar Wind Speed, km/s'],
        ['Solar Wind Speed, km/s', 'Solar Wind Density, cm^-3', 'Solar Wind Dynamic Pressure, nPa', 'Solar Wind Temperature, K'],
        [PROTON_LABEL, PROTON_LABEL, PROTON_LABEL, PROTON_LABEL]
    ]

    # Defines the grid of the plot. It has to match the feature-key matrix above.
    rows = 5
    cols = 4
    fig, ax = plt.subplots(rows, cols, figsize=(15, 18))

    for row in range(rows):
        for col in range(cols):
            sns.distplot(data[features[row][col]].replace([np.inf, -np.inf], np.nan).dropna(axis=0),
                         color="#6BC072", norm_hist=False, kde=False, ax=ax[row][col])
            ax[row][col].set_title(features[row][col])
            ax[row][col].set_xlabel(units[row][col])
            ax[row][col].set_ylabel(r"$log_{10}(Count)$")
            ax[row][col].set_yscale("log")

    fig.tight_layout()
    fig.savefig(path, dpi=300, transparent=False, tight_layout=True)


################################
# Low Level Plotting Functions #
################################
# pylint: disable = bad-whitespace
def plot_pred_obs_time(df: [pd.DataFrame], idx_range: (Any, Any) = (1000, 1200),  # noqa: C901,E252
                       path=".figures/prediction_over_time.pdf",
                       save_plot: bool = True,
                       y_scale: str = "log"):
    """ Plot predictions vs observation over time.

        Parameters:
            df: pd.DataFrame
                Data containing at least the columns ["Labels", "Predictions", "ts"].
                The column names have to match! Additionally you can provide a data frame with
                positional information, i.e. ["Labels", "Predictions", "ts", "x", "y", "z"].
            idx_range: (Any, Any)
                Represents the start and the end of the data that is used for plotting.
                Since we use time data you have to provide valid datetime objects.
            title: str
                Title of the plot
            path: str
                Path where the plot is stored. Without the file name.
            save_plot: bool
                Determines if the plot is saved.
            y_scale: str
                For defining the scale on the y axis
        Returns:
            matplotlib.Figure
    """

    fig, ax = plt.subplots(len(df), 1, figsize=(15, 4 * len(df)), sharex="all")

    if len(df) == 1:
        ax = [ax]

    # pylint: disable=consider-using-enumerate
    for i in range(len(df)):

        if isinstance(idx_range[0], int):
            df_tmp = df[i].iloc[df[i]["ts"].sort_values().index[idx_range[0]:idx_range[1]]]
        else:
            df_tmp = df[i][(df[i]["ts"] >= idx_range[0]) & (df[i]["ts"] <= idx_range[1])]

        # Transform the date back to original data space!
        # df_tmp["Predictions"] = np.log10(np.exp(df_tmp["Predictions"]))
        # df_tmp["Labels"] = np.log10(np.exp(df_tmp["Labels"]))

        # df_tmp["Predictions"] = np.log10(df_tmp["Predictions"])
        # df_tmp["Labels"] = np.log10(df_tmp["Labels"])

        sns.lineplot(df_tmp["ts"], df_tmp["Predictions"], ax=ax[i], color="Red", label="Prediction")
        sns.lineplot(df_tmp["ts"], df_tmp["Labels"], ax=ax[i], color="Blue", label="Observation")
        ax[i].set_title(f"Channel: {i + 1}")
        ax[i].set_ylabel("Proton intensities \n" +
                         r"$\frac{1}{(s \cdot cm^2 \cdot steradian \cdot keV)}$")

        locator = mdates.AutoDateLocator(minticks=3, maxticks=14)
        formatter = mdates.ConciseDateFormatter(locator)
        ax[i].xaxis.set_major_locator(locator)
        # ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax[i].xaxis.set_major_formatter(formatter)

        ax[i].set_xlabel('t in HH:MM')
        # ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=45, ha='right')
        #  ax[i].legend(loc='center left', bbox_to_anchor=(1, 0.5))

        if "x" in df_tmp.columns and "y" in df_tmp.columns and "z" in df_tmp.columns and i == 0:
            coord_ax = ax[-1].twiny()
            coord_ax.set_xlabel('Position (x, y ,z) [RE]')
            coord_ax.xaxis.set_ticks_position("bottom")
            coord_ax.xaxis.set_label_position("bottom")
            coord_ax.tick_params(direction="in")

            # Offset the twin axis below the host
            coord_ax.spines["bottom"].set_position(("axes", -0.55))
            coord_ax.spines["bottom"].set_visible(True)
            coord_ax.spines["bottom"].set_linewidth(0.5)
            coord_ax.spines["bottom"].set_color("black")

            coord_ax.set_frame_on(True)
            coord_ax.patch.set_visible(False)

            # Set number of ticks
            # ticks = df_tmp[:: int(np.ceil(len(df_tmp) / 3))]
            ticks = df_tmp[df_tmp.index % (len(df_tmp) // 3) == 0]

            coord_ax.set_xlim(ax[i].get_xlim())

            coord_ax.set_xticks(mpl.dates.date2num(ticks["ts"]))
            # coord_ax.grid(False, linestyle='-.', color='#808080')
            coord_ax.grid(None)

            tmp_pos = ("(" +
                       np.round(ticks['x'], 3).astype(str) + ", " +
                       np.round(ticks['y'], 3).astype(str) + ", " +
                       np.round(ticks['z'], 3).astype(str) + ") \n" +
                       ticks['ts'].dt.strftime('%H:%M'))

            coord_ax.set_xticklabels(tmp_pos)

        fig.subplots_adjust(hspace=0.5)

        if y_scale is not None:
            ax[i].set_yscale(y_scale)
            
    handles, labels = ax[-1].get_legend_handles_labels()

    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0, -1.5))

    plt.tight_layout()
    fig.tight_layout()

    if save_plot and path is not None:
        plt.savefig(f'{path.replace(".pdf", "")}.pdf', dpi=300)
        # plt.savefig(f'{path.replace(".pdf", "")}.png', dpi=300)

    return fig


def plot_residual_dist(y_hat: pd.DataFrame, y: pd.DataFrame, ts: pd.DataFrame,
                       channel: int = 0, title: str = "title", kde: bool = True,
                       save_plot: bool = True, dateformat: str = "%Y", path: str = "./"):
    """ Plotting the residuals between y_hat and y colored by the timestamps ts.

        Parameters:
            y_hat: pd.DataFrame
                Predictions
            y: pd.DataFrame
                Observed values corresponding to the predictions.
            ts: pd.DataFrame
                Timestamps corresponding to the predictions and observations.
            channel: int
                Channel that is used for the plots.
            name: str
                Name of the models that created those results.
            title: str
                Title of the plot.
            save_plot: bool
                Flag to decide if the plot should be saved within this function.
            dateformat: str
                Format string to decide which kind of date is used for the coloring.
        Return:
            matplotlib.Figure: Returns the figure of the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    data = pd.concat([y_hat, y, ts], axis=1)
    data.columns = ["y_hat", "y", "ts"]

    data["year"] = data["ts"].dt.year
    data["residual"] = (data["y"] - data["y_hat"])

    cmap = mpl.cm.get_cmap('summer')

    max_date = data["ts"].dt.strftime(dateformat).astype(np.int64).max()
    min_date = data["ts"].dt.strftime(dateformat).astype(np.int64).min()
    norm = mpl.colors.Normalize(vmin=min_date, vmax=max_date)

    ####################################
    # Preparation of grouped histogram #
    ####################################
    x_multi, years = [], []
    for year, group in data.sort_values("year").groupby(by='year'):
        years.append(year)
        x_multi.append(group['residual'].values)

    ##########################
    # Plot grouped histogram #
    ##########################
    # cf. https://www.weirdgeek.com/2018/11/plotting-stacked-histogram/
    # cf. https://www.weirdgeek.com/2018/11/multiple-histograms-different-length/
    plt.hist(x_multi, density=True, histtype='bar', alpha=.5)

    # Old implmententation
    # sns.distplot(x_multi, color="#309eaf", ax=ax , norm_hist=True, kde=True)

    ########################
    # Add coloring to bars #
    ########################
    num_years = len(years)
    c_idx = 0
    for i, rec in enumerate(ax.patches):
        # ts_temp = data.loc[(err > rec.xy[0]) & (err <= rec.xy[0] + rec.get_width()), "ts"]
        if (i % num_years) == 0:
            if c_idx < len(years):
                color_val = norm(int(years[c_idx]))
                c_idx += 1

        rec.set_color(cmap(color_val))

        # if(not ts_temp.empty):
        #     ts_mode = ts_temp.dt.strftime(dateformat).astype(np.int64).mode()[0]
        #     rec.set_color(cmap(norm(int(ts_mode))))

    #######
    # KDE #
    #######
    if kde:
        for year, group in data.groupby(by='year'):
            sns.distplot(a=group['residual'], label=year, hist=False, kde_kws=dict(ls='dashed'),
                         ax=ax, color=cmap(norm(int(year))))

    cb = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap), ax=ax, ticks=np.linspace(0, 1, len(years)))
    # cb.ax.set_yticklabels([data["ts"].min().strftime(dateformat),
    #                        data["ts"].max().strftime(dateformat)])
    cb.ax.set_yticklabels(years)

    ax.set_title(f"{title} - Channel: {channel + 1}")
    ax.set_xlabel(r"$y_{" + str(channel + 1) + r"} - \hat{y}_{" + str(channel + 1) + r"}$")  # noqa: E501, W605
    ax.set_ylabel("Count")

    if save_plot:
        plt.savefig(f'{path}{title}_{channel}.pdf', dpi=300)
        plt.savefig(f'{path}{title}_{channel}.png', dpi=300)

    return fig


def plot_residual_hist(y_hat: pd.DataFrame, y: pd.DataFrame,
                       channel: int = 0, name: str = "", title: str = None,
                       save_plot: bool = True):
    """ Create hisogram of the residuals.

        Parameters:
            y_hat: pd.DataFrame
                Predictions
            y: pd.DataFrame
                Corresponding ground truths
            channel: int
                Channel that we want to use for plotting
            name: str
                Name of the model leading to this results
            title: str
                Title of the plot. Also used for file name.
            save_plot: bool
                Flag to decide if the plot should be saved
        Returns:
            matplotlib.Figure
    """
    # We are already in log space
    err = (y - y_hat)

    if title is not None:
        title = f"{name} Histogramm of model residuals (Channel: {channel + 1})"

    fig, ax = plt.subplots(figsize=(10, 7))

    _ = sns.distplot(err, color="#309eaf", ax=ax, norm_hist=False, kde=False)

    # 0.1 Quantile
    ax.axvline(err.quantile(0.1), label="10% Quantile", color="#fcc544")

    # 0.9 Quantile
    ax.axvline(err.quantile(0.5), label="50% Quantile", color="#1ebd4b")

    # 0.9 Quantile
    ax.axvline(err.quantile(0.9), label="90% Quantile", color="#fc6644")

    ax.legend()

    ax.set_title(title)
    ax.set_xlabel(
        r"$log(y_{" + str(channel + 1) + r"}) - log(\hat{y}_{" + str(channel + 1) + r"})$")
    ax.set_ylabel(r"Absolute Count")

    ################
    # Save Results #
    ################
    if save_plot:
        fig.savefig(f'.results/{title}.pdf', dpi=300)
        fig.savefig(f'.results/{title}_transparent.png', dpi=300, transparent=True,
                    facecolor="none", edgecolor="white")
    return fig


def create_bin_matrix(data: pd.DataFrame, cols: [str], features: pd.DataFrame = None,
                      resolution: float = 1, aggr: str = "size", num_bins: int = None,
                      zero_centered: bool = True, symmetric: bool = True,
                      plain_bins: bool = False):
    """ This function bins the given data frame either to a 2D bin matrix or a 1D bin list.

        The most simple form to use this function is:
            bins, _, _ = create_bin_matrix(data, cols=["p1"]).
        It creates bins with resolution/width of 1 over the the feature "p1".

        For the 2D case just provide a second bin dimension, i.e.:
            bins, _, _ = create_bin_matrix(data, cols=["x", "y"])

        Important: Per default you create bins and counts the data instances within a bin.
        If you want to have some aggregation over the bin you have to provide the feature that
        you want to aggregate over and define which aggregation you want to use.

        E.g. bins, _, _ = create_bin_matrix(data, cols=["x"], feature=data["p1"], aggr="mean")
        for having the mean over channel 1 in each bin.

        Bins can be created in two different ways. First, providing a resolution, i.e. the width
        of the bin, or alternatively providing the number of bins you want to have.
        Those are mutual exclusive.

        Parameters:
            data: pd.DataFrame
                Data frame containing the data that is used for the binning.
                It has to contain at least the columns provided in cols.
            cols: [str]
                List of column names over that are the bins are created.
                ["col1"] for 1D bin, ["col1", "col2"] for 2D bin
            features: pd.DataFrame
                If you want to aggregate the data within a bin you have to provide a feature over
                that the data is aggregated.
            resolution: float (default: 1)
                Resolution/width of a bin.
            aggr: str ("size")
                Defines which kind of aggregation is applied per bin. Default is "size" which
                corresponds to the count of the data instances in a bin.
            num_bins: int
                Defines the number of bins that are created (equal width) over the data.
                Important you can use resolution OR num_bins, but not both.
            zero_centered: bool
                Creating bins that are centered around zero.
            symmetric: bool
                In case of a 2D bin matrix symmetric defines if both axis have the same bins.
                If true the larger frontier is used for creating the bins.
            plain_bins: bool
                If true the function returns the plain bins instead of the bin matrix.

        Return:
            3-tuple: plain_bins/bin_matrix, max_value, min_value
    """
    if num_bins is not None:
        resolution = 0

    if len(cols) > 1:
        max_x, max_y = np.ceil(data[cols].max(axis=0))
        min_x, min_y = np.sign(data[cols].min(axis=0)) * np.ceil(np.abs(data[cols].min(axis=0)))

        if zero_centered:
            xfrontier = (-max(max_x, np.abs(min_x)), np.ceil(max(max_x, np.abs(min_x))))
            yfrontier = (-max(max_y, np.abs(min_y)), np.ceil(max(max_y, np.abs(min_y))))
        else:
            xfrontier = (min_x, np.sign(max_x) * np.ceil(np.abs(max_x)))
            yfrontier = (min_y, np.sign(max_y) * np.ceil(np.abs(max_y)))
    else:
        max_x = np.ceil(data[cols].max(axis=0))[0]
        min_x = (np.sign(data[cols].min(axis=0)) * np.ceil(np.abs(data[cols].min(axis=0))))[0]

        if zero_centered:
            xfrontier = (-max(max_x, np.abs(min_x)), np.ceil(max(max_x, np.abs(min_x))))
        else:
            xfrontier = (min_x, np.sign(max_x) * np.ceil(np.abs(max_x)))

    if symmetric:
        yfrontier = xfrontier

    if num_bins is not None:
        # Range
        resolution = ((xfrontier[1] - xfrontier[0])) / num_bins

    ser1, _ = pd.cut(data[cols[0]], retbins=True,
                     bins=pd.interval_range(start=xfrontier[0], freq=resolution, periods=None,
                                            end=xfrontier[1] + resolution, closed='left'))
    ser1 = ser1.rename("bins_x")
    ser2 = None

    if len(cols) > 1:
        if num_bins is not None:
            # Range
            resolution = (yfrontier[1] - yfrontier[0]) / num_bins

            num_bins += 1

            # resolution = None

        ser2, _ = pd.cut(data[cols[1]], retbins=True,
                         bins=pd.interval_range(start=yfrontier[0], freq=resolution, periods=None,
                                                end=yfrontier[1] + resolution, closed='left'))
        ser2 = ser2.rename("bins_y")

    binsx = pd.concat([ser1, ser2, features], axis=1)

    if len(cols) > 1:
        bins = binsx.groupby(["bins_x", "bins_y"]).agg(aggr).reset_index()
        # bins_unstacked = bins.unstack(level=0)
        bins_unstacked = bins.pivot(
            index=bins.columns[1], columns=bins.columns[0], values=bins.columns[2])
    else:
        bins = binsx.groupby(["bins_x"]).agg(aggr).reset_index()

    if plain_bins or len(cols) == 1:
        return bins, np.nanmax(bins[bins.columns[-1]]), np.nanmin(bins[bins.columns[-1]]), ser1, ser2
    else:
        return bins_unstacked, np.nanmax(bins[bins.columns[-1]]), np.nanmin(bins[bins.columns[-1]])


def create_bin_matrix_v2(x: pd.DataFrame, y: pd.DataFrame, bins_x: Any = 20, bins_y: Any = 20):
    """ Simplified version to bin data.

        Parameters:
            x: pd.DataFrame
                Data for which the bins on the x axis are creatd in the final matrix.
            y: pd.DataFrame
                Data for which the bins on the y axis are creatd in the final matrix.
            bins_x: int or pd.IntervalIndex
                Either the number of bins that are created or a list of already created bins.
            bins_y: int or pd.IntervalIndex
                Either the number of bins that are created or a list of already created bins.
        Returns:
            pd.DataFrame, pd.IntervalIndex, pd.IntervalIndex: Returns the matrix with the binned values,
            the single bins for the x axis as well as the bins on the y axis (in this order).
    """
    bin_x_per_sample, bins_x = pd.cut(x, retbins=True, bins=bins_x)
    bin_y_per_sample, bins_y = pd.cut(y, retbins=True, bins=bins_x)

    binsx = pd.concat([bin_x_per_sample, bin_y_per_sample, x, y], axis=1).set_axis(["bins_x", "bins_y", "a", "b"], axis=1)  # noqa: E501
    binsx[["bins_x", "bins_y"]] = binsx.apply({"bins_x": lambda x: np.round(x.right, 3),
                                               "bins_y": lambda x: np.round(x.right, 3)})

    matrix = binsx.groupby(["bins_x", "bins_y"]).agg("count").reset_index()
    M = matrix.pivot(index=matrix.columns[1],
                     columns=matrix.columns[0],
                     values=matrix.columns[2]).sort_index(ascending=False)

    return M, bins_x, bins_y


def plot_simple_heatmap_v2(matrices: [pd.DataFrame, pd.DataFrame], plot_dist: bool = True, nth_tick: int = 3):
    """ Simplified version of plotting prediction heatmaps with the possibility to
        add distribution plots.

        Paramters:
            matrices: [pd.DataFrame, pd.DataFrame]
                List of data frames where both corresponds to the matrix that is plotted in the
                or the left subplot.
            plot_dist: bool
                Flag to determine if the distriubtions over the x and y axis should be plotted.
            nth_tick: int
                Number that defines the tick frequency of the plot.
        Return:
            matplotlib.figure.Figure, matplotlib.axes.Axes: Figure and axis object of the plot.
    """
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.5), sharey=True, sharex=True)

    ax[0].set_ylabel("Prediction " + PROTON_LABEL)

    for i, m in enumerate(matrices):
        ax[i].set_yticks(list(range(m.shape[0]))[::nth_tick])
        ax[i].set_yticklabels(m.index.values[::nth_tick])

        ax[i].set_xticks(range(m.shape[0])[::nth_tick])
        ax[i].set_xticklabels(m.columns.values[::nth_tick], rotation=90)

        # Plot heatmap
        im = ax[i].imshow(m, cmap=mpl.cm.get_cmap('viridis'))
        ax[i].grid(False)

        min_x, max_x = ax[i].get_xlim()
        min_y, max_y = ax[i].get_ylim()
        ax[i].plot([min_x, min_y], [max_x, max_y], ':k')

        if plot_dist:
            fig.subplots_adjust(wspace=0.3)
            xdist = ax[i].inset_axes([0, 1.07, 1, 0.15])
            ydist = ax[i].inset_axes([1.07, 0, 0.15, 1])

            ydist.barh(range(m.shape[0]), m.sum(axis=1).values[::-1])
            ydist.set_yticks(range(m.shape[0])[::-nth_tick])
            ydist.margins(0)
            ydist.set_yticklabels([])
            ydist.set_xscale("log")

            xdist.bar(range(m.shape[0]), m.sum(axis=0).values)
            xdist.set_xticks(range(m.shape[0])[::nth_tick])
            xdist.margins(0)
            xdist.set_xticklabels([])
            xdist.set_yscale("log")

    # fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([1, 0.15, 0.02, 0.81])
    cbar_ax.set_title("Counts")
    fig.colorbar(im, cax=cbar_ax)

    return fig, ax


def plot_simple_heatmap(counts: pd.DataFrame, xlabel: str = "", ylabel: str = "",  # noqa: C901
                        ax: mpl.axes.Axes = None, set_aspect: bool = False,
                        cmap: mpl.colors.Colormap = None,
                        norm: mpl.colors.Normalize = None,
                        cnorm: mpl.colors.Normalize = None,
                        cbar: bool = False, tick_freq: int = 5, freq_type: str = "val",
                        figsize: tuple = (7, 7), distr: bool = False, diag: bool = False,
                        dist_scale: str = None, title: str = None, cbar_pos="left",
                        cbar_label="Counts", annot=False, annot_kws={"size": 8}, fmt=None):
    """ Plots the heatmap.

        The most simple form to create a heatmap is just providing the data (2D matrix):
        bins, _, _ = create_bin_matrix(data, cols=["x", "y"])
        _ = plot_simple_heatmap(bins)

        Parameters:
            counts: pd.DataFrame
                Data frame contains the counts (or other values like averages) that
                reflect the value in a cell of the heatmap.
            xlable: str
                Label for the x axis.
            ylabel: str
                Label for the y axis.
            ax: matplotlib.axes.Axes (optional)
                Axis on which the plot is created.
            fig: matplotlib.figure.Figure
                Not used anymore!
            set_aspect: bool (optional)
                Determines if the plot will be represented as square, i.e. width and
                height are equal.
            cmap: matplotlib.colors.Colormap
                Color map for the heatmap itself
            norm: matplotlib.colors.Normalize
                Normalization for the coloring in the heatmap
                (see matplotlib documentation for details).
            cnorm: matplotlib.colors.Normalize
                Normalization for the coloring in the color bar
            cbar: bool
                Determines if the color bar is plotted. Per default false, since we proved the
                colorbar from outside.
            tick_freq: int
                Frequency for the ticks at each axis (x and y). E.g. 5 means every fifth
                tick is plotted.
            freq_type: str {"val", "ind"}
                Determines on what the tick frequency is applied. "val" means that we use for
                example every fifth value and "ind" means we plot every fifth index. E.g. "val"
                and tick_freq = 5 and we have the values  [0, 0.2, 0.4, 0.6, 0.8, 1, 3, 4, 5],
                we would plot the tick at value 0 and value 5. In case of "ind" we plot
                the tick at 0, 1.
            figsize: tuple
                Figsize.
            distr: bool
                Determines if the distribution should be plotted for the x and y axis.
            diag: bool
                Determines if a diagonal lines should be plotts.
            dist_scale: str
                Defines the scale of the y axis for the distribution plots, if they are provided.
            title: str
                Title of the plot.
        Return:
            Either ax or fig, counts.
    """
    cbar_axis = None
    fig = None

    if cmap is None:
        cmap = mpl.cm.get_cmap('viridis')

    if ax is None:
        ###############
        # Create Grid #
        ###############
        fig = plt.figure(figsize=figsize, constrained_layout=True)

        nrows = 2
        ncols = 3
        plot_pos = (1, 1)
        top_distr = (0, 1)
        bot_distr = (1, 2)
        cbar_ax_pos = (1, 0)
        width_ratios = [0.2, 7, 1]
        height_ratios = [1, 7]

        if cbar_pos == "top":
            nrows = 3 if distr else 2
            ncols = 2
            plot_pos = (2, 0) if distr else (1, 0)
            top_distr = (1, 0)
            bot_distr = (2, 1)
            cbar_ax_pos = (0, 0)
            width_ratios = [6, 1]
            height_ratios = [0.2, 1, 7] if distr else [0.2, 7]

        gs = GridSpec(nrows, ncols, figure=fig, width_ratios=width_ratios,
                      height_ratios=height_ratios)

        ax = fig.add_subplot(gs[plot_pos])

        if cbar:
            cbar_axis = fig.add_subplot(gs[cbar_ax_pos])

            if cnorm is None:
                cnorm = mpl.colors.Normalize(vmin=counts.min().min(),
                                             vmax=counts.max().max(), clip=False)

            orientation = "horizontal" if cbar_pos == "top" or cbar_pos == "bottom" else "vertical"

            # cb1 = mpl.colorbar.ColorbarBase(cbar_axis, cmap=cmap, norm=cnorm, orientation=orientation, pad=0.05)  # noqa: E501
            cb1 = fig.colorbar(cm.ScalarMappable(norm=cnorm, cmap=cmap),
                               cax=cbar_axis, orientation=orientation, pad=0.001)
            cb1.set_label(cbar_label)
            cbar_axis.yaxis.set_label_position('left')

        if distr:
            ax_obs = ax.get_figure().add_subplot(gs[top_distr])
            plt.setp(ax_obs.get_xticklabels(), visible=False)

            ax_pred = ax.get_figure().add_subplot(gs[bot_distr])
            plt.setp(ax_pred.get_yticklabels(), visible=False)

        if title is not None:
            fig.suptitle(title)
    else:
        if distr:
            fig = ax.get_figure()
            divider = make_axes_locatable(ax)
            ax_obs = divider.append_axes("top", "20%", pad=0.1)
            ax_pred = divider.append_axes("right", "20%", pad=0.1)

            ax_pred.get_yaxis().set_ticks([])
            ax_obs.get_xaxis().set_ticks([])

    # Heatmap
    # x and y labels doesn't match x and y in the data frame!
    _ = sns.heatmap(counts.replace(-0, 0), ax=ax, cbar=cbar and (cbar_axis is None),
                    cmap=cmap, norm=norm,
                    xticklabels=["%.2f" % x.left if hasattr(
                        x, 'left') else x for x in counts.columns],
                    yticklabels=["%.2f" % x.left if hasattr(
                        x, 'left') else x for x in counts.index],
                    annot=annot, annot_kws=annot_kws, fmt=fmt)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_xticklabels(counts.columns.map(lambda x: "%.2f" % x.left if hasattr(x, 'left') else x))
    ax.set_yticklabels(counts.index.map(lambda x: "%.2f" % x.left if hasattr(x, 'left') else x))

    if set_aspect:
        ax.set_aspect(1)

    if diag:
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        lims = [min(x0, y0), max(x1, y1)]
        ax.plot(lims, lims, ':k')

    if distr:
        # Obs. Distribution
        sns.barplot(x="bins_x", y="Counts",
                    data=counts.unstack().reset_index()[["bins_x", 0]].groupby(
                        ["bins_x"]).agg("sum").reset_index().rename(columns={0: "Counts"}),
                    color="#028166",
                    ax=ax_obs)

        ax_obs.set_xlabel("")
        ax_obs.set_ylabel("Counts\n[log]" if dist_scale == "log" else "Counts")
        ax_obs.set_facecolor("white")
        ax_obs.spines['top'].set_visible(False)
        ax_obs.spines['right'].set_visible(False)
        ax_obs.spines['bottom'].set_visible(False)
        ax_obs.spines['left'].set_visible(False)
        ax_obs.get_xaxis().set_ticks([])

        ax_obs.set_yscale(dist_scale)

        # Pred. Distribution
        sns.barplot(y="bins_y", x="Counts",
                    data=counts.unstack().reset_index()[["bins_y", 0]].groupby(
                        ["bins_y"]).agg("sum").reset_index().rename(columns={0: "Counts"}),
                    color="#028166",
                    ax=ax_pred)

        ax_pred.invert_yaxis()
        # ax_pred.set_xticklabels(ax_pred.get_xticks())
        ax_pred.set_facecolor("white")

        ax_pred.set_ylabel("")
        ax_pred.set_xlabel("Counts\n[log]" if dist_scale == "log" else "Counts")#, ha='left')
        ax_pred.spines['top'].set_visible(False)
        ax_pred.spines['right'].set_visible(False)
        ax_pred.spines['bottom'].set_visible(False)
        ax_pred.spines['left'].set_visible(False)
        ax_pred.get_yaxis().set_ticks([])

        ax_pred.set_xscale(dist_scale)

        plt.setp(ax_pred.xaxis.get_minorticklabels(), rotation=270)
        plt.setp(ax_pred.xaxis.get_majorticklabels(), rotation=270)

    num_tick_labels = len(ax.xaxis.get_ticklabels())
    for (i, xl) in enumerate(ax.xaxis.get_ticklabels()):
        visibility = True
        if freq_type == "val":
            try:
                visibility = (float(xl.get_text()) % tick_freq ==
                              0 or i == num_tick_labels - 1 or i == 0)
            except:  # noqa: E722
                visibility = (i % tick_freq == 0 or i == num_tick_labels - 1 or i == 0)
        elif freq_type == "ind":
            visibility = (i % tick_freq == 0 or i == num_tick_labels - 1 or i == 0)

        xl.set_visible(visibility)

    num_tick_labels = len(ax.yaxis.get_ticklabels())
    for (i, yl) in enumerate(ax.yaxis.get_ticklabels()):
        visibility = True
        if freq_type == "val":
            try:
                visibility = (float(yl.get_text()) % tick_freq ==
                              0 or i == num_tick_labels - 1 or i == 0)
            except:  # noqa: E722
                visibility = (i % tick_freq == 0 or i == num_tick_labels - 1 or i == 0)
        elif freq_type == "ind":
            visibility = (i % tick_freq == 0 or i == num_tick_labels - 1 or i == 0)

        yl.set_visible(visibility)

    if fig is None:
        return ax
    else:

        fig.subplots_adjust(hspace=0, wspace=0)
        fig.tight_layout()
        return fig, counts


def grouped_bar_plot(data: pd.DataFrame, swap: bool = False, ax: mpl.axes.Axes = None,
                     title: str = "title", value_label: str = "value", space_factor: float = 4.,
                     barSize: float = 25., orientation="v", figsize=(7, 15), bar_label=False,
                     error=None, val=None, pivot=None):
    """ Function to create grouped bar plots without transforming the data.

        Parameters:
        -----------
            data: pd.DataFrame
                Data you want to plot
            swap: bool
                Flag to decide if the axis should be swapped
            ax: mpl.axes.Axes
                Matplot axis if you want to add the plot to another one.
            space_factor: float
                Factor that influences the space between groups.
            barSize: float
                Determine the size of the bars, e.g. width or height depending on orientation
            orientation: str
                To determine if the bars shuld apear on horizontal or veritcal axis.
                "v" or "vertical" = Bars appear vertically
                "h" or "horizontal" = Bars appear horizontally
        Return:
        -------
            plt.Axis: Matplot axis object.
    """

    if orientation == "v" or orientation == "vertical" and ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    elif orientation == "h" or orientation == "horizontal" and ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if error is not None:
        df_error = data[["feature", error]].pivot(columns='feature', values=error)  # data[error]
    else:
        df_error = None

    if val is not None and pivot is not None:
        num_bars = len(data[pivot].unique())
        data = data[[pivot, val]].pivot(columns=pivot, values=val)
    else:
        num_bars = len(data)

    if swap:
        df_bar = data.T
        df_error = df_error.T
    else:
        df_bar = data

    num_groups = len(df_bar)
    num_bars = len(df_bar.columns)
    space = barSize * space_factor

    # pylint: disable = redefined-outer-name
    cm = plt.get_cmap('viridis')
    norm = mpl.colors.Normalize(vmin=0, vmax=num_bars, clip=False)
    # ax.set_prop_cycle('color', [cm(i//3*3.0/ (num_bars + 5 )) for i in range(num_bars + 5)])

    for i, key in enumerate(df_bar.columns):
        y_coord = (((barSize * num_bars) + space) * np.arange(num_groups) +
                   i * barSize) - ((barSize / 2 * num_bars) / 2)

        if orientation == "v" or orientation == "vertical":
            # ax = sns.barplot(x = y_coord, y = barSize,  edgecolor='white', label=key, color=cm(norm(i)), ax = ax)  # noqa: E501
            ax.barh(y_coord, width=df_bar[key], height=barSize,
                    edgecolor='white', label=key, color=cm(norm(i)))
            # ax.barh(y_coord[-1] + space, width=0, height=barSize)
        elif orientation == "h" or orientation == "horizontal":
            # ax = sns.barplot(x = df_bar[key], y = y_coord, label=key, ax = ax)
            ax.bar(y_coord, height=df_bar[key], width=barSize, edgecolor='white',
                   label=key, color=cm(norm(i)), yerr=df_error[key])
            # ax.bar(y_coord[-1] + space, height=0, width=barSize)

    if orientation == "v" or orientation == "vertical":
        ax.set_yticks(((barSize * num_bars) + space) * np.arange(num_groups))
        ax.set_yticklabels(df_bar.index)
        ax.set_ylabel(value_label)
    elif orientation == "h" or orientation == "horizontal":
        ax.set_xticks(((barSize * num_bars) + space) * np.arange(num_groups))
        ax.set_xticklabels(df_bar.index)
        ax.set_xlabel(value_label)
        ax.tick_params(axis='x', rotation=90)

    if bar_label:
        idx = 0
        for i, rect in enumerate(ax.patches):
            if i % (len(df_bar.index)) == 0 and i != 0:
                idx += 1

            if orientation == "v" or orientation == "vertical":
                width = rect.get_width()
                ax.text(rect.get_y() + rect.get_height() / 2, width + 0.01, "test",
                        ha='center', va='bottom', rotation=0)

            elif orientation == "h" or orientation == "horizontal":
                height = rect.get_height()

                ax.text(rect.get_x() + rect.get_width() / 2, height + 0.01, df_bar.columns[idx],
                        ha='center', va='bottom', rotation=90)

    ax.set_title(title)
    ax.legend()

    return fig, ax


def plot_proton_relation(data: pd.DataFrame, feature: str = "rdist", channel: str = "p1",
                         bins: pd.DataFrame = None, bin_config: dict = None, xlabel: str = "",
                         ylabel: str = "", label: str = None, ax: mpl.axes.Axes = None,
                         title: str = "title", linestyle: str = "o", automatic_ticks: bool = True):
    """ Plot relation between proton intensitiy and different features.

        Bins are per default 20 and the aggregation used for a bin is the mean.
        To adapt the binning configuration you can either provide separately created bins with
        "bins" or provide a "bin_config".

        Parameter:
            data: pd.DataFrame
                Data frame containing at least the feature your provide with "feature"
                and the channel you provide with "channel".
            feature: str
                Column name of the feature from which you want to create the distribution plot.
            channel: str
                The channel you want to use for the distribution plot (i.e. the y axis)
            bins: pd.DataFrame
                Data frame containing the bins you want to use for the distribution.
            bin_config: dict
                Configuration for the binning. This configuration corresponds to the parameter
                in the create_bin_matrix function.
                Default
                bin_config = dict(
                    aggr="mean",
                    num_bins= 20,
                    symmetric=False,
                    plain_bins= True,
                    zero_centered=False
                )
            xlabel: str
                Label for x axis.
            ylabel: str
                Label for y axis
            label: str
                Label for the data itself. E.g. when you have multiple data instances
                plotted in on one plot.
            ax: matplotlib.axes.Axes
                Axis at which the plot should be plotted. If not provided a new axis/figure
                is created.
            title: str
                Title of the plot
            linestyle: str
                Matplotlib linestyle. Can also be just a marker!
            automatic_ticks: bool
                Determines if the ticks should be made automatically or not.
        Return:
            2-tuple: ax, x-axis-ticks
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    if data.empty or all(data[feature].isna()):
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()

        return ax, None

    if bins is None:
        if bin_config is None:
            bin_config = dict(
                aggr="mean",
                num_bins=20,
                symmetric=False,
                plain_bins=True,
                zero_centered=False
            )
        else:
            bin_config = {
                dict(
                    aggr="mean",
                    num_bins=20,
                    symmetric=False,
                    plain_bins=True,
                    zero_centered=False
                ), *bin_config}
        ###############
        # Create bins #
        ###############
        bins, _, _, _, _ = create_bin_matrix(data, cols=[feature], features=data[channel], **bin_config)

    ###################################################
    # Calculate confidence interval and extract error #
    ###################################################
    y_bins = pd.cut(data[feature], bins=pd.IntervalIndex(bins["bins_x"]))
    conf_int = [mean_confidence_interval(data[channel][y_bins == x_bin])
                for x_bin in bins["bins_x"]]
    err = np.array(conf_int)[:, 1]

    ########
    # Plot #
    ########
    if ax is None:
        _, ax = plt.subplots()

    col = 0 if channel not in bins.columns else channel

    # Vertical errorbars: Confidence interval for this bin
    # Horizontal errorbars: Width of the bin (divided by two because the
    # error bar is plotted left and right)
    ax.errorbar(bins["bins_x"].map(lambda x: x.left).astype(float), bins[col],
                bins[col] - err, bins["bins_x"].map(lambda x: (x.right - x.left) / 2),
                linestyle="None", color='k', linewidth=1, zorder=1)
    if not automatic_ticks:
        ax.set_xticks(bins["bins_x"].map(lambda x: x.left).astype(float))

    ax.plot(bins["bins_x"].map(lambda x: x.left).astype(float),
            bins[col], linestyle, markersize=6, label=label, zorder=2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(title)
    ax.legend()

    return ax, bins["bins_x"].map(lambda x: x.left).astype(float)


def mean_confidence_interval(data: list, confidence: float = 0.95):
    """ Calculates mean confidence interval.
    """
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), sp.stats.sem(a)
    # pylint: disable=protected-access
    h = se * sp.stats.t._ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h
