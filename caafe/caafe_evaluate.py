import copy
import logging
from sklearn.model_selection import cross_val_score

import pandas as pd
from tabpfn import scripts
import numpy as np
from .data import get_X_y
from .preprocessing import make_datasets_numeric, make_dataset_numeric
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from autogluon.tabular import TabularPredictor
from sklearn.exceptions import NotFittedError
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor

def evaluate_dataset(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    prompt_id,
    name,
    method,
    metric_used,
    target_name,
    max_time=5,
    seed=0,
):
    df_train, df_test = copy.deepcopy(df_train), copy.deepcopy(df_test)
    df_train, _, mappings = make_datasets_numeric(
        df_train, None, target_name, return_mappings=True
    )
    df_test = make_dataset_numeric(df_test, mappings=mappings)
    if df_test is not None:
        test_x, test_y = get_X_y(df_test, target_name=target_name)
    df_train = df_train.apply(pd.to_numeric, errors='coerce')
    # Step 3: Handle missing values
    df_train.fillna(0, inplace=True)

    x, y = get_X_y(df_train, target_name=target_name)
    feature_names = list(df_train.drop(target_name, axis=1).columns)
    np.random.seed(0)
    if method == "autogluon" or method == "autosklearn2":
        if method == "autogluon":
            from tabpfn.scripts.tabular_baselines import autogluon_metric

            clf = autogluon_metric
        elif method == "autosklearn2":
            from scripts.tabular_baselines import autosklearn2_metric

            clf = autosklearn2_metric
        metric, ys, res = clf(
            x, y, test_x, test_y, feature_names, metric_used, max_time=max_time
        )  #
    elif type(method) == str:
        if method == "gp":
            from scripts.tabular_baselines import gp_metric

            clf = gp_metric
        elif method == "knn":
            from scripts.tabular_baselines import knn_metric

            clf = knn_metric
        elif method == "xgb":
            from scripts.tabular_baselines import xgb_metric

            clf = xgb_metric
        elif method == "catboost":
            from scripts.tabular_baselines import catboost_metric

            clf = catboost_metric
        elif method == "random_forest":
            from scripts.tabular_baselines import random_forest_metric

            clf = random_forest_metric
        elif method == "logistic":
            from scripts.tabular_baselines import logistic_metric

            clf = logistic_metric
        metric, ys, res = clf(
            x,
            y,
            test_x,
            test_y,
            [],
            metric_used,
            max_time=max_time,
            no_tune={},
        )
    # If sklearn classifier
    elif isinstance(method, RandomForestRegressor):
        method.fit(X=x, y=y.long())
        ys = method.predict(test_x)
    elif isinstance(method, XGBRegressor):
        method.fit(X=x, y=y)
        ys = method.predict(test_x)
    elif isinstance(method, BaseEstimator):
        method.fit(X=x, y=y.long())
        ys = method.predict_proba(test_x)
    elif isinstance(method, TabularPredictor):
        try:
            check_is_fitted(method)
            method2 = TabularPredictor(label=target_name,eval_metric='r2', verbosity=0)
            method2.fit(df_train, time_limit=max_time)
            ys = method2.predict(df_test)
        except NotFittedError:
            method.fit(df_train, time_limit=max_time)
            ys = method.predict(df_test)
    else:
        metric, ys, res = method(
            x,
            y,
            test_x,
            test_y,
            [],
            metric_used,
        )
    test_y = df_test[target_name]
    r2 = r2_score(test_y, ys) # r2 score used for regression
    logging.info("r2: {}".format(r2))
    #acc = scripts.tabular_metrics.accuracy_metric(test_y, ys) # used for classification
    #rmse = mean_squared_error(test_y, ys, squared=False)
    rmse = 0
    method_str = method if type(method) == str else "transformer"
    return {
        "r2": r2,
        "rmse": rmse,
        "prompt": prompt_id,
        "seed": seed,
        "name": name,
        "size": len(df_train),
        "method": method_str,
        "max_time": max_time,
        "feats": len(df_train.columns) - 1,
    }


def get_leave_one_out_importance(
    df_train, df_test, ds, method, metric_used, max_time=30
):
    """Get the importance of each feature for a dataset by dropping it in the training and prediction."""
    res_base = evaluate_dataset(
        ds,
        df_train,
        df_test,
        prompt_id="",
        name=ds[0],
        method=method,
        metric_used=metric_used,
        max_time=max_time,
    )

    importances = {}
    for feat_idx, feat in enumerate(set(df_train.columns)):
        if feat == ds[4][-1]:
            continue
        df_train_ = df_train.copy().drop(feat, axis=1)
        df_test_ = df_test.copy().drop(feat, axis=1)
        ds_ = copy.deepcopy(ds)

        res = evaluate_dataset(
            ds_,
            df_train_,
            df_test_,
            prompt_id="",
            name=ds[0],
            method=method,
            metric_used=metric_used,
            max_time=max_time,
        )
        importances[feat] = (round(res_base["roc"] - res["roc"], 3),)
    return importances
