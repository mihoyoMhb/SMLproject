# scripts/hyperparameter_tuning.py
import warnings

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, f1_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


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
                class_weight='balanced'
            )
            f1 = cross_val_score(
                rf, X_train, y_train.ravel(), cv=cv,
                scoring=make_scorer(f1_score, average='binary', pos_label=1),
            )
            recall = cross_val_score(
                rf, X_train, y_train.ravel(), cv=cv,
                scoring=make_scorer(recall_score, average='binary', pos_label=1),
            )
            f1_scores[i, j] = np.mean(f1)
            recall_scores[i, j] = np.mean(recall)
    return f1_scores, recall_scores


def tune_random_forest_rs(X_train, y_train, cv=5, scoring='f1', n_iter=300):
    num_pos = np.sum(y_train == 1)
    num_neg = np.sum(y_train == 0)
    ratio = num_neg / num_pos
    param_dist = {
        'n_estimators': [10 * i for i in range(5, 75)],  # 100 到 375，步长25
        'max_depth': list(range(1, 20)),  # 7 到 13
        'min_samples_split': list(range(1, 32)),
        'min_samples_leaf': list(range(1, 32)),
        'max_features': ['sqrt', 'log2', 0.1, 0.2, 0.3, 0.4, 0.5],
        'bootstrap': [True, False],
        'class_weight': ['balanced', {0: 1, 1: ratio}, {0: 1 / ratio, 1: 1}],
        'criterion': ['gini', 'entropy']
    }

    rf = RandomForestClassifier(random_state=42)
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        random_state=42
    )
    random_search.fit(X_train, y_train.ravel())
    return random_search.best_estimator_


def tune_adaboost_rs(X_train, y_train, cv=5, scoring='f1', n_iter=300):
    param_dist = {
        'n_estimators': [10 * i for i in range(10, 101)],  # 100 到 1000，步长10
        'learning_rate': np.linspace(0.01, 2, 20),  # 从 0.01 到 2 等间距选择 20 个值
        'algorithm': ['SAMME'],  # Adaboost 的算法选择
    }

    ada = AdaBoostClassifier(random_state=42)
    random_search = RandomizedSearchCV(
        estimator=ada,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        random_state=42
    )
    random_search.fit(X_train, y_train.ravel())
    return random_search.best_estimator_


def tune_knn_rs(X_train, y_train, cv=5, scoring='f1', n_iter=300):
    param_dist = {
        'n_neighbors': list(range(1, 51)),  # 从 1 到 50 的邻居数
        'weights': ['uniform', 'distance'],  # 权重选择
        'metric': ['euclidean', 'manhattan', 'minkowski'],  # 距离度量
    }

    knn = KNeighborsClassifier()
    random_search = RandomizedSearchCV(
        estimator=knn,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        random_state=42
    )
    random_search.fit(X_train, y_train.ravel())
    return random_search.best_estimator_


