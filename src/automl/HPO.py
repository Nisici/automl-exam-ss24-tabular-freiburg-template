import logging
import time

from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from ray.tune.schedulers import HyperBandScheduler
from ray import train, tune
from ray.train import Checkpoint, ScalingConfig, Result
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
from sklearn.svm import SVR
import os
import torch
import pickle
import tempfile
import xgboost as xgb
import numpy as np
from run import MODELS
from matplotlib import pyplot as plt


UNKNOWN_MODEL_MSG = "Model unknown. Choose from the following models: {}".format(MODELS)

def plot_scores(analysis, path):
    # Extract data (adjust as needed)
    df = analysis.dataframe()

    # Plot mean metric over trials
    plt.plot(df['mean_r2'])  # Replace 'metric_name' with your metric
    plt.xlabel('Trial ID')
    plt.ylabel('Mean r2')
    plt.title('Mean Metric Value per Trial')
    plt.savefig(os.path.join(path, str(time.time())+'.png'))
    plt.show()

"""
Fit algorithm until time is up or when found best hyperparameters
Returns: model, performance
"""
def choose_best_hyperparameters(x, y,plot_path, model_name):
    logging.info("Finding best hyperparameters for model")
    def train_model(config):
        model = None
        if model_name == 'xgboost':
            model = XGBRegressor(**config)
        else:
            assert UNKNOWN_MODEL_MSG
        # Perform cross-validation
        features, labels = x, y
        score = cross_val_score(model, features.to_numpy(), labels.to_numpy(), cv=5, scoring='r2')
        mean_score = np.mean(score)
        metrics = {"mean_r2" : mean_score}
        train.report(metrics)
        return metrics

    search_space = define_search_space(model_name)
    hyperband_scheduler = HyperBandScheduler(max_t=200, reduction_factor=3)
    analysis = tune.run(
        train_model,
        config=search_space,
        scheduler=hyperband_scheduler,
        num_samples=25,  # Number of hyperparameter configurations to try
        resources_per_trial={'cpu': 6},
        metric='mean_r2', mode='max',
    )
    best_result = analysis.best_result
    results = analysis.results
    logging.info("RESULTS: {}".format(results))
    logging.info("Best configuration: {}".format(analysis.best_config))
    logging.info("Best performance: {}".format(best_result))
    return analysis.best_config, best_result

"""
Returns the configuration search space based on the model type
types of models: {xgboost}
"""
def define_search_space(model_name):
    search_space = None
    if model_name == 'xgboost':
        search_space = {
            "max_depth": tune.randint(3, 10),
            "min_child_weight": tune.uniform(1, 10),
            "gamma": tune.loguniform(1e-3, 1e-1),
            "subsample": tune.uniform(0.6, 0.9),
            "colsample_bytree": tune.uniform(0.6, 0.9),
            "learning_rate": tune.loguniform(1e-3, 1e-1),
            "tree_method": tune.choice(["auto", "exact", "approx", "hist"]),
            "grow_policy": tune.choice(["depthwise", "lossguide"]),
            "lambda": tune.loguniform(1e-3, 1e-1),
            "alpha": tune.loguniform(1e-3, 5e-1),
        }
    else:
        assert UNKNOWN_MODEL_MSG
    return search_space