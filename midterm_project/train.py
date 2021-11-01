#!/usr/bin/env python
# coding: utf-8
import os
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor


def get_pipeline(model):
    categorical_features = ['smoker', 'sex', 'region']

    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    preprocessor = ColumnTransformer(
        transformers=[('cat', categorical_transformer, categorical_features)])

    pipe = Pipeline(
        steps=[('preprocessor', preprocessor), ('model', model())]
    )
    return pipe


def train(X_train, y_train):
    tree_pipe = get_pipeline(DecisionTreeRegressor)

    search = GridSearchCV(tree_pipe, params_tree, cv=num_cv, refit=True, verbose=0, n_jobs=1,
                          scoring='neg_mean_squared_error')
    search.fit(X_train, y_train)

    with open(model_file_name, 'wb') as f_out:
        pickle.dump(search.best_estimator_, f_out)

    print(f'the model is saved to {model_file_name}')


if __name__ == '__main__':
    # parameters
    num_cv = 4
    model_file_name = os.environ.get('MODEL_FILE_NAME', 'best_model.bin')
    params_tree = [
        {
            'model__max_depth': [4, 5, 6, 8],
            'model__min_samples_leaf': [10, 20, 25]
        }
    ]
    # data preparation
    data = pd.read_csv('insurance.csv')
    df_train, df_test = train_test_split(data, test_size=0.2, random_state=1, stratify=data['smoker'])
    df_train['charges'] = np.log1p(df_train['charges'])
    df_test['charges'] = np.log1p(df_test['charges'])
    y_train = df_train.pop('charges')
    y_test = df_test.pop('charges')

    train(df_train, y_train)



