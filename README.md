# Prediction of soft proton intensities in the near-Earth space using machine learning

Data: [![Data](https://zenodo.org/badge/DOI/10.5281/zenodo.4718561.svg)](https://doi.org/10.5281/zenodo.4718561)
Model Weights: [![Model Weights](https://zenodo.org/badge/DOI/10.5281/zenodo.4593065.svg)](https://doi.org/10.5281/zenodo.4593065)

This repository contains the code accompanying the paper

__Prediction of soft proton intensities in the near-Earth space using machine learning__  
_Elena A. Kronberg, Tanveer Hannan, Jens Huthmacher, Marcus Munzer, Florian Peste, Ziyang Zhou, Max Berrendorf, Evgeniy Faerman, Fabio Gastaldello, Simona Ghizzardi, Philippe Escoubet, Stein Haaland, Artem Smirnov, Nithin Sivadas, Robert C. Allen, Andrea Tiengo, and Raluca Ilie_

## Installation
Even when it is not required, it is beneficial to use virtual python environments. Hence, feel free to set up an environment and activate it before isnstall all necessary pacakeges.

```bash
(base) user: ./$ python3.8 -m venv ./env_name
(base) user: ./$ source ./env_name/bin/activate
(env_name) user: ./$
```

To use the deep horizon framework one have to prepare and set up the execution envrionment by installing the necessary dependencies. Please install all needed packages by the following command.
```bash
(env_name) user: ./$ pip install -r requirements.txt
```

## MLFlow
In order to track results to a MLFlow server, start it first by running

`mlflow server`

_Note: When storing the result for many configurations, we recommend to setup a database backend following the [instructions](https://mlflow.org/docs/latest/tracking.html)_. For the following examples, we assume that the server is running at

`http://localhost:5000`

If the MLFlow server is running at another address one can pass it as additional paramter as follows:

``` bash
(base) user: [...] --tracking AddressOfMLFLowServer
```

## Obtain data
Download the dataset from [here](https://zenodo.org/record/4718561), and copy the files `RAPID_OMNI_ML_023_traincut.h5`, `RAPID_OMNI_ML_023_testcut.h5` and `RAPID_OMNI_ML_023_robusttranscut.pkl` to the folder `/data`.

## Run HPO
For all experiments the results are logged to the running MLFlow instance.

_Note: The hyperparameter searches takes a significant amount of time (~multiple days). You can abort the script at any time, and inspect the current results via the web interface of MLFlow._

``` bash
(base) user: ./src$ python main.py --hpo "HPO"
```

We also provide the weights of the best model we found [here](https://zenodo.org/record/4593065).

## Create Plots
One can use the predefined plotting function to create the plots displayed in the paper. For simply creating all plots at once, just use the CLI as follows:

``` bash
(base) user: ./src$ python main.py --create_plots PATH/TO/MODEL_DATA/
```

For creating selected plots or plot it more interactively you can use [this](./src/visualization/create_plots.ipynb) prepared Jupyter notebook 
