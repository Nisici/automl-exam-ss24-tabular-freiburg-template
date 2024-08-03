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
        y = y.loc[x.index]
        best_model = XGBRegressor(**configuration)
        best_model.fit(x, y)
    else:
        assert HPO.UNKNOWN_MODEL_MSG
    return best_model

def write_txt(filename, raw_text):
    if not os.path.exists(filename):
        with open(filename, "x") as f:
            f.write(raw_text)
    else:
        with open(filename, "w") as f:
            f.write(raw_text)
def test_model(model, x_test, y_test, output_path, model_name):
    #test_preds = model.predict(x_test)
    logger.info("Writing performance to disk")
    if y_test is not None:
        #r2_test = r2_score(y_test, test_preds)
        r2_test = model.score(x_test, y_test)
        logger.info(f"R^2 on test set: {r2_test}")
        now = datetime.datetime.now()
        date_string = now.strftime("%Y-%m-%d %H:%M:%S")
        message = "score: {}, model: {}, time: {}\n".format(r2_test, model, date_string)
        file_path = os.path.join(output_path, "r2_score.txt")
        if not os.path.exists(file_path):
            write_txt(file_path, message)
        else:
            f = open(file_path, "a")
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
def caafe(best_model, train_merged, label_name, task_name, best_config=None):
    from caafe import CAAFEClassifier
    import openai
    def read_txt(filename):
        with open(filename, 'r') as f:
            content = f.read()
        return content
    caafe_dir = os.path.join(os.getcwd(), 'caafe', task_name)
    """
    openai.apy_key = 'YOUR-API-KEY'
    """
    dataset_description = read_txt(os.path.join(caafe_dir, 'dataset_description.txt'))
    if best_config is not None:
        logging.info("Passing the best hyperparameters configuration to caafe")
        best_config['enable_categorical'] = True
        base_model = XGBRegressor(**best_config)
    else:
        logging.info("passing best model to caafe")
        base_model = best_model
    model = CAAFEClassifier(
        base_classifier=base_model,
        llm_model="gpt-4",
        iterations=10,
        n_splits=5,
        n_repeats=1,
    )
    model.fit_pandas(train_merged, target_column_name=label_name, dataset_description=dataset_description)
    write_txt(caafe_dir + '/generated_code.txt', model.code)
    return model

def apply_caafe_code(df):
    """
    BOHB BIKE SHARING
    """

    # Feature name and description: 'is_night'
    # Usefulness: This feature will indicate if it is night time or not. This is useful because bike rentals might be lower during the night.
    # Input samples: 'hour': [23.0, 5.0, 14.0]
    df['is_night'] = df['hour'].apply(lambda x: 1 if (x > 20 or x < 6) else 0)

    # Feature name and description: 'is_winter'
    # Usefulness: This feature will indicate if it is winter or not. This is useful because bike rentals might be lower during winter due to cold weather.
    # Input samples: 'month': [6.0, 1.0, 4.0]
    df['is_winter'] = df['month'].apply(lambda x: 1 if (x == 12 or x < 3) else 0)

    # Feature name and description: 'peak_hours'
    # Usefulness: This feature will indicate if the current hour is a peak hour or not. This is useful because bike rentals might be higher during peak hours (e.g., commuting hours).
    # Input samples: 'hour': [23.0, 5.0, 14.0]
    df['peak_hours'] = df['hour'].apply(lambda x: 1 if (x >= 7 and x <= 9) or (x >= 17 and x <= 19) else 0)
    """
    BOHB BRAZILIAN HOUSES
    # Feature name: 'tax_per_area'
    # Usefulness: This feature gives the property tax per unit area. It can help in understanding the tax efficiency of the house.
    # Input samples: 'area': [45.0, 278.0, 280.0], 'property_tax_(BRL)': [8.0, 155.0, 1126.0]
    df['tax_per_area'] = df['property_tax_(BRL)'] / df['area']  # Feature name: 'parking_per_room'
    # Usefulness: This feature gives the ratio of parking spaces to rooms. It can help in understanding the convenience of the house.
    # Input samples: 'parking_spaces': [0.0, 4.0, 2.0], 'rooms': [2.0, 4.0, 3.0]
    df['parking_per_room'] = df['parking_spaces'] / df['rooms']

    # Dropping 'parking_spaces' and 'rooms' as they are now represented in the 'parking_per_room' feature
    df.drop(columns=['parking_spaces', 'rooms'], inplace=True)
    """
    """
    THESE FEATURES WORK WITH HYPERBAND for exam_dataset
    # (House_age_and_renovation)
    # Usefulness: Combining information about the age of the house and whether it has been renovated can capture the history of the property, which can affect its price.
    # Input samples: 'yr_built': [1970.0, 1995.0, 1992], 'yr_renovated': [0.0, 0.0, 0.0]
    df['house_age_and_renovation'] = 2022 - df['yr_built'] + df['yr_renovated']

    # Drop 'yr_built' and 'yr_renovated' columns as they are redundant after creating 'house_age_and_renovation'
    df.drop(columns=['yr_built', 'yr_renovated'], inplace=True)

    # (Living_area_ratio)
    # Usefulness: The ratio between the living room area and the total house area can provide insights into the layout and functionality of the house.
    # Input samples: 'sqft_living': [2090.0, 1450.0, 3020.0], 'sqft_lot': [7416.0, 5175.0, 360241.0]
    df['living_area_ratio'] = df['sqft_living'] / (df['sqft_living'] + df['sqft_lot'])

    # (City_proximity_score)
    # Usefulness: Creating a proximity score based on latitude and longitude can capture the relative location of the house.
    # Input samples: 'lat': [47.41, 47.71, 47.27], 'long': [-122.18, -122.34, -122.09]
    df['city_proximity_score'] = df['lat'] + df['long']
    # (Distance_from_city_center)
    # Usefulness: Distance of the house from the city center can be a significant factor in determining house prices, as proximity to the city center often affects property values.
    # Input samples: 'lat': [47.41, 47.71, 47.27], 'long': [-122.18, -122.34, -122.09]
    df['distance_from_city_center'] = ((df['lat'] - 47.6) ** 2 + (df['long'] + 122.3) ** 2) ** 0.5
    # Explanation why the column 'sqft_basement' is dropped
    df.drop(columns=['sqft_basement'], inplace=True)
    # Explanation why the column 'date_month' is dropped
    df.drop(columns=['date_month'], inplace=True)

    # Explanation why the column 'date_day' is dropped
    df.drop(columns=['date_day'], inplace=True)
    # Feature name: 'bed_bath_ratio'
    # Usefulness: The ratio of bedrooms to bathrooms can provide an insight into the layout of the house, which could be a factor influencing the price.
    # Input samples: 'bedrooms': [4.0, 3.0, 3.0], 'bathrooms': [1.75, 2.5, 1.75]
    df['bed_bath_ratio'] = df['bedrooms'] / df['bathrooms']

    # Explanation why the column 'bedrooms' and 'bathrooms' are dropped
    # The 'bedrooms' and 'bathrooms' features are now represented in the 'bed_bath_ratio' feature, making them redundant.
    df.drop(columns=['bedrooms', 'bathrooms'], inplace=True)
    """
    return df

"""
Ecnode categorical features into integers
"""
def preprocessing(df):
    from sklearn.preprocessing import LabelEncoder
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])
    df = df.astype(float)
    return df

def make_folder(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

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
    seed = 42
    dataset = Dataset.load(datadir=DATADIR, task=task, fold=fold)
    x_train, y_train, x_test, y_test = dataset.X_train, dataset.y_train, dataset.X_test, dataset.y_test
    """
        DATA PREPROCESSING
    """
    working_dir = os.getcwd()
    if feat_eng == 'caafe':
        make_folder(os.path.join(working_dir, 'caafe_feat'))
        caafe_dir = os.path.join(working_dir, 'caafe_feat', task)
        make_folder(caafe_dir)
        assert os.path.exists(os.path.join(caafe_dir, 'dataset_description.txt')), "You have to create a dataset_description.txt with the description of the dataset"
    if task == 'exam_dataset':
        exam_path = os.path.join(working_dir, 'data', 'exam_dataset', '1')
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
            copy_of_exam_test.drop(['price'], axis=1), copy_of_exam_test['price'], test_size=0.1, random_state=seed)
    """
    if not test_caafe_feat:
        x_train = apply_caafe_code(x_train)
        x_test = apply_caafe_code(x_test)
    """
    logger.info(x_train.columns)
    x_train = preprocessing(x_train)
    x_test = preprocessing(x_test)
    y_train = y_train.loc[x_train.index]
    y_test = y_test.loc[x_test.index]

    logger.info("X TRAIN AFTER PREPROCESSING: {} \n columns: {}".format(x_train, x_train.columns))
    plot_path = os.path.join(os.getcwd(), "plots")
    output_path_without = os.path.join(output_path, 'scores', task, 'without', model)
    output_path_feat_eng = os.path.join(output_path, 'scores', task, feat_eng, model)
    best_configuration = None
    """
                AUTOML OPTIMIZATION without feature engineering
    """
    logger.info("Fitting AutoML without feature engineering")
    logger.info("Chosen model: {}".format(model))
    plot_path_no_feat_eng = os.path.join(plot_path, "without_feat_eng")
    best_configuration, best_performance = HPO.choose_best_hyperparameters(x_train, y_train,
                                                                           plot_path=plot_path_no_feat_eng,
                                                                           model_name=model, seed=seed)
    best_model = train_best_model(model, best_configuration, x_train, y_train)
    test_acc = test_model(best_model, x_test, y_test, output_path_without, model_name=model)
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
        model = caafe(best_model, train_merged, label_name, task)
        test_acc_feat_eng = test_model(model, x_test, y_test, output_path_feat_eng, model_name=model)
        logger.info("Test accuracy after feat eng: {}".format(test_acc_feat_eng))
    elif test_caafe_feat:
        output_path_feat_eng = os.path.join(output_path, 'scores', task, 'caafe', model)
        x_train = apply_caafe_code(x_train)
        x_test = apply_caafe_code(x_test)
        x_train = preprocessing(x_train)
        x_test = preprocessing(x_test)
        y_train = y_train.loc[x_train.index]
        y_test = y_test.loc[x_test.index]
        best_configuration, best_performance = HPO.choose_best_hyperparameters(x_train, y_train,
                                                                               plot_path=None,
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
