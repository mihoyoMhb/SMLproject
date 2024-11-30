# scripts/hyperparameter_tuning.py

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
# import xgboost as xgb
# from lightgbm import LGBMClassifier
# from catboost import CatBoostClassifier
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, f1_score, recall_score


# Function to perform hyperparameter tuning for Random Forest
def tune_random_forest(X_train, y_train, cv=5, scoring='f1', n_iter=50):
    num_pos = np.sum(y_train == 1)
    num_neg = np.sum(y_train == 0)
    ratio = num_neg / num_pos

    param_grid = {
        'n_estimators': [100, 300, 500, 700, 1000],
        'max_depth': [None, 5, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4, 6],
        'max_features': ['sqrt', 'log2', None],  # 移除了 'auto'
        'bootstrap': [True, False],
        'class_weight': ['balanced', {0: 1, 1: ratio}],
        'criterion': ['gini', 'entropy']
    }

    rf = RandomForestClassifier(random_state=42)
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        random_state=42
    )
    random_search.fit(X_train, y_train.ravel())
    return random_search.best_estimator_


def evaluate_parameters_rf(X_train, y_train, n_estimators_list, max_depth_list, cv=5):
    f1_scores = np.zeros((len(max_depth_list), len(n_estimators_list)))
    recall_scores = np.zeros((len(max_depth_list), len(n_estimators_list)))
    num_pos = np.sum(y_train == 1)
    num_neg = np.sum(y_train == 0)
    ratio = num_neg / num_pos
    for i, max_depth in enumerate(max_depth_list):
        for j, n_estimators in enumerate(n_estimators_list):
            print(f"Evaluating max_depth={max_depth}, n_estimators={n_estimators}")
            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                class_weight=['balanced', {0: 1, 1: ratio}]
            )
            f1 = cross_val_score(
                rf, X_train, y_train.ravel(), cv=cv,
                scoring=make_scorer(f1_score, average='binary')
            )
            recall = cross_val_score(
                rf, X_train, y_train.ravel(), cv=cv,
                scoring=make_scorer(recall_score, average='binary')
            )
            f1_scores[i, j] = np.mean(f1)
            recall_scores[i, j] = np.mean(recall)
    return f1_scores, recall_scores


# Function to perform hyperparameter tuning for XGBoost
# def tune_xgboost(X_train, y_train, cv=5, scoring='f1'):
#     num_pos = np.sum(y_train == 1)
#     num_neg = np.sum(y_train == 0)
#     ratio = num_neg / num_pos
#
#     param_grid = {
#         'n_estimators': [100, 300, 500],
#         'max_depth': [3, 5, 7],
#         'learning_rate': [0.01, 0.05, 0.1],
#         'subsample': [0.8, 1.0],
#         'colsample_bytree': [0.8, 1.0],
#         'scale_pos_weight': [ratio]
#     }
#     xgb_clf = xgb.XGBClassifier(
#         objective='binary:logistic',
#         random_state=42,
#         eval_metric='logloss'
#     )
#     grid_search = GridSearchCV(
#         estimator=xgb_clf,
#         param_grid=param_grid,
#         cv=cv,
#         scoring=scoring,
#         n_jobs=-1
#     )
#     grid_search.fit(X_train, y_train.ravel())
#     return grid_search.best_estimator_
#
#
# # Function to perform hyperparameter tuning for LightGBM
# def tune_lightgbm(X_train, y_train, cv=5, scoring='f1'):
#     num_pos = np.sum(y_train == 1)
#     num_neg = np.sum(y_train == 0)
#     ratio = num_neg / num_pos
#
#     param_grid = {
#         'n_estimators': [100, 300, 500],
#         'max_depth': [-1, 5, 10],
#         'learning_rate': [0.01, 0.05, 0.1],
#         'num_leaves': [31, 50, 100],
#         'subsample': [0.8, 1.0],
#         'class_weight': ['balanced', None],
#         'scale_pos_weight': [ratio]
#     }
#     lgbm_clf = LGBMClassifier(
#         random_state=42
#     )
#     grid_search = GridSearchCV(
#         estimator=lgbm_clf,
#         param_grid=param_grid,
#         cv=cv,
#         scoring=scoring,
#         n_jobs=-1
#     )
#     grid_search.fit(X_train, y_train.ravel())
#     return grid_search.best_estimator_
#
#
# # Function to perform hyperparameter tuning for CatBoost
# def tune_catboost(X_train, y_train, cv=5, scoring='f1'):
#     num_pos = np.sum(y_train == 1)
#     num_neg = np.sum(y_train == 0)
#     class_weights = [1, num_neg / num_pos]
#
#     param_grid = {
#         'iterations': [100, 300, 500],
#         'depth': [3, 5, 7],
#         'learning_rate': [0.01, 0.05, 0.1],
#         'l2_leaf_reg': [1, 3, 5],
#         'border_count': [32, 50, 100],
#         'class_weights': [class_weights]
#     }
#     cat_clf = CatBoostClassifier(
#         random_seed=42,
#         verbose=0
#     )
#     grid_search = GridSearchCV(
#         estimator=cat_clf,
#         param_grid=param_grid,
#         cv=cv,
#         scoring=scoring,
#         n_jobs=-1
#     )
#     grid_search.fit(X_train, y_train.ravel())
#     return grid_search.best_estimator_
