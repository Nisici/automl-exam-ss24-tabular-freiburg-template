"""An example run file which loads in a dataset from its files
and logs the R^2 score on the test set.

In the example data you are given access to the y_test, however
in the test dataset we will provide later, you will not have access
to this and you will need to output your predictions for X_test
to a file, which we will grade using github classrooms!
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from automl.data import Dataset
import argparse

import logging
from src.automl import HPO
import os
import pandas as pd
from sklearn.metrics import r2_score
import xgboost as xgb

logger = logging.getLogger(__name__)

FILE = Path(__file__).absolute().resolve()
DATADIR = FILE.parent / "data"
MODELS = ["xgboost"]

def load_and_combine_data(base_folder_path, num_folds=10):
    X_all = []
    y_all = []
    y_train = None
    print(base_folder_path)
    for i in range(1, num_folds + 1):
        X_train_path = os.path.join(base_folder_path, f'{i}', 'X_train.parquet')
        X_val_path = os.path.join(base_folder_path, f'{i}', 'X_test.parquet')
        y_train_path = os.path.join(base_folder_path, f'{i}', 'y_train.parquet')
        y_val_path = os.path.join(base_folder_path, f'{i}', 'y_test.parquet')

        X_train = pd.read_parquet(X_train_path)
        X_val = pd.read_parquet(X_val_path)
        y_train = pd.read_parquet(y_train_path)
        y_val = pd.read_parquet(y_val_path)
        X_all.append(X_train)
        X_all.append(X_val)
        y_all.append(y_train)
        y_all.append(y_val)
    label = y_train.columns[0]
    X_all = pd.concat(X_all).reset_index(drop=True)
    y_all = pd.concat(y_all)
    y_all = y_all[label]
    return X_all, y_all
"""
fit and return the model given the best configuration
"""
def train_best_model(model_name, configuration, x, y):
    best_model = None
    if model_name == 'xgboost':
        dmatrix = xgb.DMatrix(x, label=y, enable_categorical=True)
        best_model = xgb.train(
            params=configuration,
            dtrain=dmatrix,
        )
    else:
        assert HPO.UNKNOWN_MODEL_MSG
    return best_model

def test_model(model, x_test, y_test, output_path, model_name):
    if model_name == 'xgboost':
        x_test = xgb.DMatrix(x_test, enable_categorical=True)
    test_preds = model.predict(x_test)
    logger.info("Writing predictions to disk")
    with output_path.open("wb") as f:
        np.save(f, test_preds)
    if y_test is not None:
        r2_test = r2_score(y_test, test_preds)
        logger.info(f"R^2 on test set: {r2_test}")
    else:
        # This is the setting for the exam dataset, you will not have access to y_test
        logger.info("No test set")

def main(
    task: str,
    fold: int,
    output_path: Path,
    seed: int,
    datadir: Path,
    model: str
):
    """
        DATA LOADING
    """
    dataset = Dataset.load(datadir=DATADIR, task=task, fold=fold)
    x_train, y_train, x_test, y_test = dataset.X_train, dataset.y_train, dataset.X_test, dataset.y_test
    storage_path = os.path.join(os.getcwd(), "saved_models")
    """
        AUTOML OPTIMIZATION without feature engineering
    """
    logger.info("Fitting AutoML without feature engineering")
    logger.info("Chosen model: {}".format(model))
    logger.info("Best model will be saved in path: {}".format(storage_path))
    best_configuration, best_performance = HPO.choose_best_hyperparameters(x_train, y_train, storage_path=storage_path, model_name=model)
    best_model = train_best_model(model, best_configuration, x_train, y_train)
    test_model(best_model, x_test, y_test, output_path, model_name=model)
    """
           AUTOML OPTIMIZATION with feature engineering
    """
    logger.info("Fitting AutoML with feature engineering")
    #x_train, y_train = feature_engineering()
    best_configuration, best_performance = HPO.choose_best_hyperparameters(x_train, y_train, storage_path=storage_path,
                                                                           model_name=model)
    best_model = train_best_model(model, best_configuration, x_train, y_train)
    test_model(best_model, x_test, y_test, output_path, model_name=model)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task",
        type=str,
        required=False,
        help="The name of the task to run on.",
        choices=["y_prop", "bike_sharing", "brazilian_houses"]
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("predictions.npy"),
        help=(
            "The path to save the predictions to."
            " By default this will just save to './predictions.npy'."
        )
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=1,
        help=(
            "The fold to run on."
            " You are free to also evaluate on other folds for your own analysis."
            " For the test dataset we will only provide a single fold, fold 1."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed for reproducibility if you are using and randomness,"
            " i.e. torch, numpy, pandas, sklearn, etc."
        )
    )

    parser.add_argument(
        "--datadir",
        type=Path,
        default=DATADIR,
        help=(
            "The directory where the datasets are stored."
            " You should be able to mostly leave this as the default."
        )
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Whether to log only warnings and errors."
    )
    parser.add_argument(
        "--model-name",
        default='xgboost',
        help="Model name to do HPO on."
    )
    args = parser.parse_args()

    if not args.quiet:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    logger.info(
        f"Running task {args.task}"
        f"\n{args}"
    )

    main(
        task=args.task,
        fold=args.fold,
        output_path=args.output_path,
        datadir=args.datadir,
        seed=args.seed,
        model=args.model_name
    )
