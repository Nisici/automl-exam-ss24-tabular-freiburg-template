import copy
import logging

import numpy as np
from sklearn.preprocessing import LabelEncoder

import openai
from sklearn.model_selection import RepeatedKFold
from .caafe_evaluate import (
    evaluate_dataset,
)
from .run_llm_code import run_llm_code
from sklearn.model_selection import cross_val_score
from .preprocessing import make_datasets_numeric, make_dataset_numeric

import pandas as pd
def get_prompt(
    df, ds, iterative=1, data_description_unparsed=None, samples=None, **kwargs
):
    how_many = (
        "up to 10 useful columns. Generate as many features as useful for downstream classifier, but as few as necessary to reach good performance."
        if iterative == 1
        else "exactly one useful column"
    )
    return f"""
The dataframe `df` is loaded and in memory. Columns are also named attributes.
Description of the dataset in `df` (column dtypes might be inaccurate):
"{data_description_unparsed}"

Columns in `df` (true feature dtypes listed here, categoricals encoded as int):
{samples}
    
This code was written by an expert datascientist working to improve predictions. It is a snippet of code that adds new columns to the dataset.
Number of samples (rows) in training dataset: {int(len(df))}
    
This code generates additional columns that are useful for a downstream classification algorithm (such as XGBoost) predicting \"{ds[4][-1]}\".
Additional columns add new semantic information, that is they use real world knowledge on the dataset. They can e.g. be feature combinations, transformations, aggregations where the new column is a function of the existing columns.
The scale of columns and offset does not matter. Make sure all used columns exist. Follow the above description of columns closely and consider the datatypes and meanings of classes.
This code also drops columns, if these may be redundant and hurt the predictive performance of the downstream classifier (Feature selection). Dropping columns may help as the chance of overfitting is lower, especially if the dataset is small.
The classifier will be trained on the dataset with the generated columns and evaluated on a holdout set. The evaluation metric is accuracy. The best performing code will be selected.
Added columns can be used in other codeblocks, dropped columns are not available anymore.

Code formatting for each added column:
```python
# (Feature name and description)
# Usefulness: (Description why this adds useful real world knowledge to classify \"{ds[4][-1]}\" according to dataset description and attributes.)
# Input samples: (Three samples of the columns used in the following code, e.g. '{df.columns[0]}': {list(df.iloc[:3, 0].values)}, '{df.columns[1]}': {list(df.iloc[:3, 1].values)}, ...)
(Some pandas code using {df.columns[0]}', '{df.columns[1]}', ... to add a new column for each row in df)
```end

Code formatting for dropping columns:
```python
# Explanation why the column XX is dropped
df.drop(columns=['XX'], inplace=True)
```end

Each codeblock generates {how_many} and can drop unused columns (Feature selection).
Each codeblock ends with ```end and starts with "```python"
Codeblock:
"""


# Each codeblock either generates {how_many} or drops bad columns (Feature selection).


def build_prompt_from_df(ds, df, iterative=1):
    data_description_unparsed = ds[-1]
    feature_importance = {}  # xgb_eval(_obj)

    samples = ""
    df_ = df.head(10)
    for i in list(df_):
        # show the list of values
        nan_freq = "%s" % float("%.2g" % (df[i].isna().mean() * 100))
        s = df_[i].tolist()
        if str(df[i].dtype) == "float64":
            s = [round(sample, 2) for sample in s]
        samples += (
            f"{df_[i].name} ({df[i].dtype}): NaN-freq [{nan_freq}%], Samples {s}\n"
        )

    kwargs = {
        "data_description_unparsed": data_description_unparsed,
        "samples": samples,
        "feature_importance": {
            k: "%s" % float("%.2g" % feature_importance[k]) for k in feature_importance
        },
    }

    prompt = get_prompt(
        df,
        ds,
        data_description_unparsed=data_description_unparsed,
        iterative=iterative,
        samples=samples,
    )

    return prompt


def generate_features(
    ds,
    df,
    model="gpt-3.5-turbo",
    just_print_prompt=False,
    iterative=1,
    metric_used=None,
    iterative_method="logistic",
    display_method="markdown",
    n_splits=5,
    n_repeats=1,
):
    def format_for_display(code):
        code = code.replace("```python", "").replace("```", "").replace("<end>", "")
        return code

    if display_method == "markdown":
        from IPython.display import display, Markdown

        display_method = lambda x: display(Markdown(x))
    else:

        display_method = print

    assert (
        iterative == 1 or metric_used is not None
    ), "metric_used must be set if iterative"

    prompt = build_prompt_from_df(ds, df, iterative=iterative)

    if just_print_prompt:
        code, prompt = None, prompt
        return code, prompt, None

    def generate_code(messages):
        if model == "skip":
            return ""

        completion = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            stop=["```end"],
            temperature=0.5,
            max_tokens=500,
        )
        code = completion["choices"][0]["message"]["content"]
        code = code.replace("```python", "").replace("```", "").replace("<end>", "")
        return code

    def execute_and_evaluate_code_block(full_code, code):
        old_r2, old_rmse, r2, rmse = [], [], [], []
        ss = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
        df_train = df

        # Remove target column from df_train
        target_train = df_train.iloc[:, -1]
        #target_valid = df_valid.iloc[:, -1]
        df_train = df_train.drop(df_train.columns[-1], axis=1)
        #df_valid = df_valid.drop(df_valid.columns[-1], axis=1)

        df_train_extended = copy.deepcopy(df_train)
        #df_valid_extended = copy.deepcopy(df_valid)
        try:
            df_train = run_llm_code(
                full_code,
                df_train,
                convert_categorical_to_integer=not ds[0].startswith("kaggle"),
            )
            """
            df_valid = run_llm_code(
                full_code,
                df_valid,
                convert_categorical_to_integer=not ds[0].startswith("kaggle"),
            )
            """
            df_train_extended = run_llm_code(
                full_code + "\n" + code,
                df_train_extended,
                convert_categorical_to_integer=not ds[0].startswith("kaggle"),
            )
            """
            df_valid_extended = run_llm_code(
                full_code + "\n" + code,
                df_valid_extended,
                convert_categorical_to_integer=not ds[0].startswith("kaggle"),
            )
            """

        except Exception as e:
            display_method(f"Error in code execution. {type(e)} {e}")
            display_method(f"```python\n{format_for_display(code)}\n```\n")
            return e, None, None, None, None
        logging.info(full_code + "\n" + code)
        # Add target column back to df_train
        df_train = make_dataset_numeric(df_train, None)
        df_train_extended = make_dataset_numeric(df_train_extended, None)
        x_old = df_train
        x_extended = df_train_extended
        index_to_keep_ext = x_extended.index
        index_to_keep_old = x_old.index
        # Select corresponding rows in y
        y_old = target_train.loc[index_to_keep_old]
        y_extended = target_train.loc[index_to_keep_ext]

        """"
        df_train[ds[4][-1]] = target_train
        df_valid[ds[4][-1]] = target_valid
        df_train_extended[ds[4][-1]] = target_train
        df_valid_extended[ds[4][-1]] = target_valid
        """
        from contextlib import contextmanager
        import sys, os
        with open(os.devnull, "w") as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            try:
                result_old = cross_val_score(iterative_method, x_old.to_numpy(), y_old.to_numpy(), cv=5, scoring='r2', n_jobs=6)
                result_extended = cross_val_score(iterative_method, x_extended.to_numpy(), y_extended.to_numpy(), cv=5, scoring='r2', n_jobs=6)
            finally:
                sys.stdout = old_stdout
            old_r2 = np.mean(result_old)
            old_rmse = 0
            r2 = np.mean(result_extended)
            rmse = 0
            """
            old_r2 += [result_old["r2"]]
            old_rmse += [result_old["rmse"]]
            r2 += [result_extended["r2"]]
            rmse += [result_extended["rmse"]]
            """
        return None, r2, rmse, old_r2, old_rmse

    messages = [
        {
            "role": "system",
            "content": "You are an expert datascientist assistant solving Kaggle problems. You answer only by generating code. Answer as concisely as possible.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]
    display_method(f"*Dataset description:*\n {ds[-1]}")

    n_iter = iterative
    full_code = ""

    i = 0
    while i < n_iter:
        try:
            code = generate_code(messages)
        except Exception as e:
            display_method("Error in LLM API." + str(e))
            continue
        i = i + 1
        logging.info("i: {}".format(i))
        e, r2, rmse, old_r2, old_rmse = execute_and_evaluate_code_block(
            full_code, code
        )
        if e is not None:
            messages += [
                {"role": "assistant", "content": code},
                {
                    "role": "user",
                    "content": f"""Code execution failed with error: {type(e)} {e}.\n Code: ```python{code}```\n Generate next feature (fixing error?):
                                ```python
                                """,
                },
            ]
            continue

        # importances = get_leave_one_out_importance(
        #    df_train_extended,
        #    df_valid_extended,
        #    ds,
        #    iterative_method,
        #    metric_used,
        # )
        # """ROC Improvement by using each feature: {importances}"""

        improvement_r2 = np.nanmean(r2) - np.nanmean(old_r2)
        improvement_rmse = np.nanmean(rmse) - np.nanmean(old_rmse)

        add_feature = True
        add_feature_sentence = "The code was executed and changes to ´df´ were kept."
        if improvement_r2 < 0:
            add_feature = False
            add_feature_sentence = f"The last code changes to ´df´ were discarded. (Improvement: {improvement_r2})"

        display_method(
            "\n"
            + f"*Iteration {i}*\n"
            + f"```python\n{format_for_display(code)}\n```\n"
            + f"Performance before adding features R2 {np.nanmean(old_r2):.3f}, RMSE {np.nanmean(old_rmse):.3f}.\n"
            + f"Performance after adding features R2 {np.nanmean(r2):.3f}, RMSE {np.nanmean(rmse):.3f}.\n"
            + f"Improvement R2 {improvement_r2:.3f}, RMSE {improvement_rmse:.3f}.\n"
            + f"{add_feature_sentence}\n"
            + f"\n"
        )

        if len(code) > 10:
            messages += [
                {"role": "assistant", "content": code},
                {
                    "role": "user",
                    "content": f"""Performance after adding feature R2 {np.nanmean(r2):.3f}, RMSE {np.nanmean(rmse):.3f}. {add_feature_sentence}
Next codeblock:
""",
                },
            ]
        if add_feature:
            full_code += code

    return full_code, prompt, messages
