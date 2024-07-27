"""An example run file which loads in a dataset from its files
and logs the R^2 score on the test set.

In the example data you are given access to the y_test, however
in the test dataset we will provide later, you will not have access
to this and you will need to output your predictions for X_test
to a file, which we will grade using github classrooms!
"""
from __future__ import annotations
import pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score
import numpy as np
from automl.data import Dataset
from automl.automl import AutoML, write_txt, read_txt
import argparse
import logging
from openfe import OpenFE, transform, tree_to_formula
import os
from autofeat import FeatureSelector, AutoFeatRegressor
from matplotlib import pyplot as plt


logger = logging.getLogger(__name__)

FILE = Path(__file__).absolute().resolve()
DATADIR = FILE.parent / "data"

"""
Does feature engineering of dataset x (pd.Dataframe) with labels y, returning a new pd.Dataframe with engineered 
features.
We use OpenFE
"""
def feature_engineering(x_train, y_train, x_test, caafe=False):
    # we don't do feature engineering with this function but we use caafe
    if caafe:
        return x_train, y_train, x_test
    n_jobs = 4
    ofe = OpenFE()
    ofe.fit(data=x_train, label=y_train, n_jobs=n_jobs)  # generate new features
    x_train, x_test = transform(x_train, x_test, ofe.new_features_list[:10],
                                n_jobs=n_jobs)  # transform the train and test data according to generated features.
    for feature in ofe.new_features_list[:10]:
        logger.info(tree_to_formula(feature))
    return x_train, x_test

def save_dataset(x_train, x_test, y_train, y_test, filepath, i=1):
    x_train.to_parquet(os.path.join(filepath, str(i), 'X_train.parquet'))
    x_test.to_parquet(os.path.join(filepath, str(i), 'X_test.parquet'))
    y_train.to_parquet(os.path.join(filepath, str(i), 'y_train.parquet'))
    y_test.to_parquet(os.path.join(filepath, str(i), 'y_test.parquet'))

def main(
    task: str,
    fold: int,
    output_path: Path,
    seed: int,
    datadir: Path,
    feat_eng: str
):
    dataset = Dataset.load(datadir=datadir, task=task, fold=fold)
    logger.info("Fitting AutoML")
    # You do not need to follow this setup or API it's merely here to provide
    # an example of how your automl system could be used.
    # As a general rule of thumb, you should **never** pass in any
    # test data to your AutoML solution other than to generate predictions.
    X_train, X_test, y_train = dataset.X_train, dataset.X_test, dataset.y_train
    if feat_eng == 'openfe':
        # y_prop test score = 0.07
        X_train, X_test = feature_engineering(dataset.X_train, dataset.y_train, dataset.X_test)
        automl = AutoML(seed=seed, label_name=dataset.y_train.name, presets='good_quality', time_limit=30)
    elif feat_eng == 'caafe':
        automl = AutoML(seed=seed, label_name=dataset.y_train.name, automl_name='caafe', path=os.getcwd())
    elif feat_eng == 'autofeat':
        steps = 1
        logger.info("X_train before: {}".format(X_train.head))
        automl = AutoML(seed=seed, label_name=dataset.y_train.name)
        automl.fit(X_train, y_train)
        test_preds: np.ndarray = automl.predict(X_test)
        r2_test = r2_score(dataset.y_test, test_preds)
        logger.info("Test score before: {}".format(r2_test))
        afreg = AutoFeatRegressor(verbose=1, feateng_steps=steps)
        X_train_trans = afreg.fit_transform(X_train, y_train)
        logger.info("X_train after: {}".format(X_train_trans.head))
        automl = AutoML(seed=seed, label_name=dataset.y_train.name)
        automl.fit(X_train_trans, y_train)
        X_test_trans = afreg.transform(X_test)
        test_preds: np.ndarray = automl.predict(X_test_trans)
        r2_test = r2_score(dataset.y_test, test_preds)
        logger.info(f"R^2 on test set: {r2_test}")

        save_dataset(X_train, X_test, y_train.to_frame(), dataset.y_test.to_frame(), filepath=datadir)
        logger.info("saved dataset with new engineered features, try an AutoML method on this new dataset.")
        return
    else:
        automl = AutoML(seed=seed, label_name=dataset.y_train.name)
    automl.fit(X_train, y_train)
    test_preds: np.ndarray = automl.predict(X_test)

    # Write the predictions of X_test to disk
    # This will be used by github classrooms to get a performance
    # on the test set.
    logger.info("Writing predictions to disk")
    with output_path.open("wb") as f:
        np.save(f, test_preds)

    if dataset.y_test is not None:
        r2_test = r2_score(dataset.y_test, test_preds)
        logger.info(f"R^2 on test set: {r2_test}")
    else:
        # This is the setting for the exam dataset, you will not have access to y_test
        logger.info(f"No test set for task '{task}'")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="The name of the task to run on.",
        choices=["y_prop", "bike_sharing", "brazilian_houses", "bike_sharing_autofeat"]
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
        "--feat-eng",
        type=str,
        default=None,
        choices=["caafe", "openfe", "autofeat"],
        help=(
            "Choose the type of feature engineering to apply to the dataset"
        )
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
        feat_eng=args.feat_eng
    )
