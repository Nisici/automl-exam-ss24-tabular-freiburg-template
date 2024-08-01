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
from scikeras.wrappers import KerasRegressor

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

def build_mlp(input_dim, num_layers, hidden_size, learning_rate):
    import tensorflow as tf
    from tensorflow.keras.losses import MeanSquaredError
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.metrics import R2Score
    model = Sequential()
    model.add(Dense(hidden_size, activation='relu', input_shape=(input_dim,)))  # Input layer

    for _ in range(num_layers - 1):
        model.add(Dense(hidden_size, activation='relu'))

    model.add(Dense(1))  # Output layer

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss=MeanSquaredError(), optimizer=optimizer, metrics=[R2Score()])
    return model
"""
Fit algorithm until time is up or when found best hyperparameters
Returns: model, performance
"""
def choose_best_hyperparameters(x, y, plot_path, model_name):
    logging.info("Finding best hyperparameters for model")
    def train_model(config):
        model = None
        if model_name == 'xgboost':
            model = XGBRegressor(**config)
        elif model_name == 'mlp':
            from tensorflow.keras.callbacks import EarlyStopping
            input_dim = len(x.columns)
            model = build_mlp(input_dim, config["num_layers"],
                              config["hidden_size"], config["learning_rate"])
            early_stop = EarlyStopping(patience=10, restore_best_weights=True)
            model = KerasRegressor(build_fn=model, verbose=0)
            model.fit(x, y, epochs=100, batch_size=128, validation_split=0.2, callbacks=[early_stop])

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
        num_samples=10,  # Number of hyperparameter configurations to try
        resources_per_trial={'cpu': 6},
        metric='mean_r2', mode='max',
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
            "n_estimators": tune.randint(80, 500),
            "max_depth": tune.randint(3, 12),
            "min_child_weight": tune.uniform(1, 10),
            "gamma": tune.loguniform(1e-3, 1e-1),
            "subsample": tune.uniform(0.6, 1),
            "colsample_bytree": tune.uniform(0.6, 0.9),
            "learning_rate": tune.loguniform(1e-3, 5e-1),
            "lambda": tune.loguniform(1e-3, 1),
            "alpha": tune.loguniform(1e-3, 1),
        }
    elif model_name == 'mlp':
        search_space = {
            "num_layers": tune.choice([2, 3, 4]),  # Choose between 2, 3, or 4 layers
            "hidden_size": tune.randint(16, 128),  # Integer between 16 and 128
            "learning_rate": tune.loguniform(1e-5, 1e-2),  # Logarithmic scale from 1e-5 to 1e-2
            # Add other hyperparameters you want to tune
        }
    else:
        assert UNKNOWN_MODEL_MSG
    return search_space