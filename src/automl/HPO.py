import logging

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

UNKNOWN_MODEL_MSG = "Model unknown. Choose from the following models: {}".format(MODELS)

"""
Fit algorithm until time is up or when found best hyperparameters
Returns: model, performance
"""
def choose_best_hyperparameters(x, y, storage_path, model_name):
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
        # Report the mean accuracy to Ray Tune
        return {"mean_r2": mean_score}

    search_space = define_search_space(model_name)
    hyperband_scheduler = HyperBandScheduler(max_t=200, reduction_factor=3)
    analysis = tune.run(
        train_model,
        config=search_space,
        scheduler=hyperband_scheduler,
        num_samples=20,  # Number of hyperparameter configurations to try
        resources_per_trial={'cpu': 6},
        metric='mean_r2', mode='max',
        keep_checkpoints_num=1,  # Keep only the best checkpoint based on mean_r2
        checkpoint_score_attr='mean_r2',  # Use r2 to determine best checkpoint
        storage_path=storage_path
    )
    best_result = analysis.best_result
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
            "n_estimators": tune.randint(100, 1000),
            "tree_method": tune.choice(["auto", "exact", "approx", "hist"]),
            "grow_policy": tune.choice(["depthwise", "lossguide"]),
            "lambda": tune.loguniform(1e-3, 1e-1),
            "alpha": tune.loguniform(1e-3, 1e-1),
        }
    else:
        assert UNKNOWN_MODEL_MSG
    return search_space