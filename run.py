"""An example run file which loads in a dataset from its files
and logs the R^2 score on the test set.

In the example data you are given access to the y_test, however
in the test dataset we will provide later, you will not have access
to this and you will need to output your predictions for X_test
to a file, which we will grade using github classrooms!
"""
from __future__ import annotations

import datetime
from math import atan2

from xgboost import XGBRegressor

import time
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
from openfe import OpenFE, tree_to_formula, transform, TwoStageFeatureSelector
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
FILE = Path(__file__).absolute().resolve()
DATADIR = FILE.parent / "data"
MODELS = ["xgboost", "mlp"]

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
        x = preprocessing(x)
        best_model = XGBRegressor(**configuration)
        best_model.fit(x, y)
    else:
        assert HPO.UNKNOWN_MODEL_MSG
    return best_model

def test_model(model, x_test, y_test, output_path, model_name):
    test_preds = model.predict(x_test)
    logger.info("Writing performance to disk")
    if y_test is not None:
        r2_test = r2_score(y_test, test_preds)
        logger.info(f"R^2 on test set: {r2_test}")
        file_path = os.path.join(output_path, "r2_score.txt")
        f = open(file_path, "a")
        now = datetime.datetime.now()
        date_string = now.strftime("%Y-%m-%d %H:%M:%S")
        message = "score: {}, model: {}, time: {}\n".format(r2_test, model, date_string)
        f.write(message)
        f.close()
    else:
        # This is the setting for the exam dataset, you will not have access to y_test
        logger.info("No test set")
    return r2_test
def remove_underscores_and_parenthesis(strings):
    new_strings = []
    for s in strings:
        s = s.replace("_", "")
        s = s.replace("(", "")
        s = s.replace(")", "")
        new_strings.append(s)
    return new_strings
#add best_n features found by the feature method
def feature_engineering(x_train, y_train, x_test, n_jobs=6, feat_eng='openfe', best_n=5, only_select=False):
    x_train.columns = remove_underscores_and_parenthesis(x_train.columns)
    x_test.columns = remove_underscores_and_parenthesis(x_test.columns)
    if feat_eng == 'openfe':
        if only_select:
            fs = TwoStageFeatureSelector(n_jobs=n_jobs)
            features = fs.fit(data=x_train, label=y_train)
            logger.info("Openfe selected: {} features".format(len(features)))
            x_train = x_train[features[:best_n]]
            x_test = x_test[features[:best_n]]
        else:
            ofe = OpenFE()
            new_features = ofe.fit(data=x_train, label=y_train, n_jobs=n_jobs,) # generate new features
            x_train, x_test, = transform(x_train, x_test, new_features[:best_n],
                                         n_jobs=n_jobs)
    return x_train, x_test
"""
Use the configuration given by the automl method without feature engineering to create a new model,
this model gets trained using CAAFE and write the new features into a file
"""
def caafe(train_merged, label_name, task_name, best_config=None):
    from caafe import CAAFEClassifier
    import openai
    def read_txt(filename):
        with open(filename, 'r') as f:
            content = f.read()
        return content

    def write_txt(filename, raw_text):
        with open(filename, "w") as f:
            f.write(raw_text)
    caafe_dir = os.path.join(os.getcwd(), 'caafe', task_name)
    openai.api_key = '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'
    dataset_description = read_txt(os.path.join(caafe_dir, 'dataset_description.txt'))
    if best_config is not None:
        base_model = XGBRegressor(**best_config)
    else:
        base_model = XGBRegressor()
    model = CAAFEClassifier(
        base_classifier=base_model,
        llm_model="gpt-3.5-turbo",
        iterations=5
    )
    model.fit_pandas(train_merged, target_column_name=label_name, dataset_description=dataset_description)
    write_txt(caafe_dir + '/generated_code.txt', model.code)
    return model

def apply_caafe_code(df):

    # (Total_sqft: total square footage of the house)
    # Usefulness: Total_sqft can capture the overall size of the house, which is an important factor in determining the price.
    # Input samples: 'sqft_living': [2090.0, 1450.0, 3020.0], 'sqft_lot': [7416.0, 5175.0, 360241.0], 'sqft_above': [1050.0, 1030.0, 3020.0]
    df['total_sqft'] = df['sqft_living'] + df['sqft_lot'] + df['sqft_above']

    # (Age_of_house: number of years since the house was built)
    # Usefulness: Age_of_house can provide insight into the property's condition and potential impact on the price.
    # Input samples: 'yr_built': [1970.0, 1995.0, 1992.0], 'date_year': [2014.0, 2014.0, 2014.0]
    df['age_of_house'] = df['date_year'] - df['yr_built']
    # (Renovated: binary indicator if the house has been renovated)
    # Usefulness: Renovated can capture whether a house has been renovated, which can influence the price significantly.
    # Input samples: 'yr_renovated': [0.0, 0.0, 0.0]
    df['renovated'] = df['yr_renovated'].apply(lambda x: 1 if x > 0 else 0)
    # Explanation why the column 'yr_built', 'yr_renovated' are dropped
    df.drop(columns=['yr_built', 'yr_renovated'], inplace=True)

    return df
"""
Make all dataset numeric and replace NaN values with zeros.
Drop columns which have only 1 unique value 
"""
def preprocessing(df):
    df = df.apply(pd.to_numeric, errors='coerce')
    df.fillna(0, inplace=True)
    nunique = df.nunique()
    cols_to_drop = nunique[nunique == 1].index
    df.drop(cols_to_drop, axis=1)
    return df

def main(
    task: str,
    fold: int,
    output_path: Path,
    seed: int,
    datadir: Path,
    model: str,
    feat_eng: str,
    test_caafe_feat: bool
):
    """
        DATA LOADING
    """
    dataset = Dataset.load(datadir=DATADIR, task=task, fold=fold)
    x_train, y_train, x_test, y_test = dataset.X_train, dataset.y_train, dataset.X_test, dataset.y_test
    """
        DATA PREPROCESSING
    """
    x_train = preprocessing(x_train)
    x_test = preprocessing(x_test)
    if task == 'exam_dataset':
        exam_path = os.path.join(os.getcwd(), 'data', 'exam_dataset', '1')
        x_test = pd.read_parquet(os.path.join(exam_path, 'X_test.parquet'))
        test_data = pd.read_csv(os.path.join(exam_path, 'kc_house_data.csv'))
        test_data['date_year'] = test_data['date'].str[0:4].astype(int)
        test_data['date_month'] = test_data['date'].str[4:6].astype(int)
        test_data['date_day'] = test_data['date'].str[6:8].astype(int)
        # Drop the original 'date' column
        test_data = test_data.drop(columns=['date'])
        cols = list(x_test.columns)
        cols.append('price')
        copy_of_exam_test = test_data[cols]
        x_train, x_test, y_train, y_test = train_test_split(
            copy_of_exam_test.drop(['price'], axis=1), copy_of_exam_test['price'], test_size=0.1, random_state=42)
    plot_path = os.path.join(os.getcwd(), "plots")
    output_path_without = os.path.join(output_path, 'scores', task, 'without', model)
    output_path_feat_eng = os.path.join(output_path, 'scores', task, feat_eng, model)
    """
                AUTOML OPTIMIZATION without feature engineering
    """
    """
    logger.info("Fitting AutoML without feature engineering")
    logger.info("Chosen model: {}".format(model))
    plot_path_no_feat_eng = os.path.join(plot_path, "without_feat_eng")
    best_configuration, best_performance = HPO.choose_best_hyperparameters(x_train, y_train,
                                                                           plot_path=plot_path_no_feat_eng,
                                                                           model_name=model)
    best_model = train_best_model(model, best_configuration, x_train, y_train)
    test_acc = test_model(best_model, x_test, y_test, output_path_without, model_name=model)
    """
    """
           AUTOML OPTIMIZATION with feature engineering
    """

    if feat_eng == 'openfe':
        logger.info("Fitting AutoML with feature engineering: {}".format(feat_eng))
        plot_path_feat_eng = os.path.join(plot_path, "feat_eng")
        if task == 'y_prop':
            x_train, x_test = feature_engineering(x_train, y_train, x_test, feat_eng=feat_eng, best_n= 20, only_select=True)
        else:
            x_train, x_test = feature_engineering(x_train, y_train, x_test, feat_eng=feat_eng)
        best_configuration, best_performance = HPO.choose_best_hyperparameters(x_train, y_train, plot_path=plot_path_feat_eng,
                                                                               model_name=model)
        best_model = train_best_model(model, best_configuration, x_train, y_train)
        test_acc_feat_eng = test_model(best_model, x_test, y_test, output_path_feat_eng, model_name=model)
        logger.info("Test accuracy after feat eng: {}".format(test_acc_feat_eng))
    elif feat_eng == 'caafe':
        logger.info("Fitting AutoML with feature engineering: {}".format(feat_eng))
        label_name = y_train.name
        train_merged = x_train.merge(y_train, left_index=True, right_index=True)
        model = caafe(train_merged, label_name, task)
        test_acc_feat_eng = test_model(model, x_test, y_test, output_path_feat_eng, model_name=model)
        logger.info("Test accuracy after feat eng: {}".format(test_acc_feat_eng))
    elif test_caafe_feat:
        output_path_feat_eng = os.path.join(output_path, 'scores', task, 'caafe', model)
        x_train = apply_caafe_code(x_train)
        x_test = apply_caafe_code(x_test)
        best_configuration, best_performance = HPO.choose_best_hyperparameters(x_train, y_train,
                                                                               plot_path=plot_path_no_feat_eng,
                                                                               model_name=model)
        best_model = train_best_model(model, best_configuration, x_train, y_train)
        test_acc_caafe = test_model(best_model, x_test, y_test, output_path_feat_eng, model_name=model)
        logger.info("R2 test score with caafe features: {}".format(test_acc_caafe))
    logger.info("Test accuracy without feat eng: {}".format(test_acc))
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task",
        type=str,
        required=False,
        help="The name of the task to run on.",
        choices=["y_prop", "bike_sharing", "brazilian_houses", "exam_dataset"]
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=os.getcwd(),
        help=(
            "The path to save the predictions to."
            " By default this will just save to the current working directory."
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
        help="Model name to do HPO on.",
        choices=MODELS
    )
    parser.add_argument(
        "--feat-eng",
        default="without",
        help="Choose the feature engineering, default is without feature engineering",
        choices= ['without', 'openfe', 'caafe']
    )
    parser.add_argument(
        "--test-caafe-feat",
        default=False,
        help="Test caafe features by applying caafe code to dataset",
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
        model=args.model_name,
        feat_eng= args.feat_eng,
        test_caafe_feat=args.test_caafe_feat
    )
