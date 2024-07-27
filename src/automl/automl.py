"""AutoML class for regression tasks.

This module contains an example AutoML class that simply returns dummy predictions.
You do not need to use this setup or sklearn and you can modify this however you like.
"""
from __future__ import annotations
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
import logging
from autogluon.tabular import TabularPredictor
from automl.data import Dataset
from caafe import CAAFEClassifier
import openai
import os

logger = logging.getLogger(__name__)

METRICS = {"r2": r2_score}

def read_txt(filename):
    with open(filename, 'r') as f:
        content = f.read()
    return content

def write_txt(filename, raw_text):
    with open(filename, "w") as f:
        f.write(raw_text)

class AutoML:

    def __init__(
        self,
        seed: int,
        label_name: str,
        metric: str = "r2",
        automl_name: str = 'autogluon', # autogluon, auto_sklearn or TabPFN (if classification)
        path: str = None,
        presets = 'medium_quality',
        time_limit=5
    ) -> None:
        self.seed = seed
        self.metric = METRICS[metric]
        self._model = TabularPredictor(label=label_name,eval_metric='r2', verbosity=2)
        self.automl_name = automl_name
        self.presets = presets
        if automl_name == 'caafe':
            #self.base_model = RandomForestRegressor()
            self._model = TabularPredictor(label=label_name,eval_metric='r2', verbosity=0)
            self.base_model = self._model
            self._model = CAAFEClassifier(
                base_classifier=self.base_model,
                llm_model="gpt-3.5-turbo",
                iterations=10
            )
        self.label_col = label_name
        self.path = path
        self.time_limit = time_limit
        """
        if automl_name == 'autogluon':
            self._model = TabularPredictor(hyperparameter_tune=True, label=label_name)
        elif automl_name == 'auto-sklearn':
            pass
        """

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> AutoML:
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            random_state=self.seed,
            test_size=0.2,
        )
        if X_train.isnull().any().any() or y_train.isnull().any():
            print("Missing values detected! Please handle them before training.")
            return
        train_merged = Dataset.merge(X_train, y_train)
        if self.automl_name == 'autogluon':
            self._model.fit(train_merged, time_limit=self.time_limit, presets=self.presets)
            y_preds = self._model.predict(X_val)
            val_score = self._model.evaluate_predictions(y_val, y_preds)
        elif self.automl_name == 'caafe':
            caafe_dir = os.path.join(self.path, 'caafe')
            dataset_description = read_txt(caafe_dir + '/dataset_description.txt')
            self._model.fit_pandas(train_merged, target_column_name=self.label_col, dataset_description=dataset_description)
            write_txt(caafe_dir + '/generated_code.txt', self._model.code)
            print('FINISHED CAAFE FITTING')
            y_preds = self._model.predict(X_val)
            val_score = self.base_model.evaluate_predictions(y_val, y_preds)

        #logger.info(f"Validation score: {val_score:.4f}")
        logger.info("Validation score: {}".format(val_score))
        return self


    def predict(self, X: pd.DataFrame) -> np.ndarray:
        #previous code
        """"
        if self._model is None:
            raise ValueError("Model not fitted")

        return self._model.predict(X)  # type: ignore
        """
        return self._model.predict(X)
