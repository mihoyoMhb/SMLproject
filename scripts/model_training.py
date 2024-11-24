# scripts/model_training.py

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
import xgboost as xgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


# Function to train Logistic Regression
def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train.ravel())
    return model


# Function to train Linear Discriminant Analysis (LDA)
def train_lda(X_train, y_train):
    model = LinearDiscriminantAnalysis()
    model.fit(X_train, y_train.ravel())
    return model


# Function to train Quadratic Discriminant Analysis (QDA)
def train_qda(X_train, y_train):
    model = QuadraticDiscriminantAnalysis()
    model.fit(X_train, y_train.ravel())
    return model


# Function to train K-Nearest Neighbors (KNN)
def train_knn(X_train, y_train, n_neighbors=5):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train.ravel())
    return model


# Function to train Decision Tree Classifier
def train_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train.ravel())
    return model


# Function to train Random Forest Classifier
def train_random_forest(X_train, y_train, n_estimators=100):
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train.ravel())
    return model


# Function to train Bagging Classifier (using Decision Tree as the base estimator)
def train_bagging(X_train, y_train, n_estimators=100):
    model = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train.ravel())
    return model


# Function to train AdaBoost Classifier
def train_adaboost(X_train, y_train, n_estimators=100):
    model = AdaBoostClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train.ravel())
    return model


# Function to train Gradient Boosting Classifier
def train_gradient_boosting(X_train, y_train, n_estimators=100):
    model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train.ravel())
    return model


def train_xgboost(X_train, y_train, n_estimators=100):
    num_pos = (y_train == 1).sum()
    num_neg = (y_train == 0).sum()
    ratio = num_neg / num_pos

    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=n_estimators,
        random_state=42,
        scale_pos_weight=ratio,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train.ravel())
    return model


def train_lightgbm(X_train, y_train, n_estimators=100):
    num_pos = (y_train == 1).sum()
    num_neg = (y_train == 0).sum()
    ratio = num_neg / num_pos

    model = LGBMClassifier(
        n_estimators=n_estimators,
        random_state=42,
        class_weight='balanced',
        scale_pos_weight=ratio  # 或者只使用class_weight='balanced'
    )
    model.fit(X_train, y_train.ravel())
    return model


def train_catboost(X_train, y_train, n_estimators=100):
    num_pos = (y_train == 1).sum()
    num_neg = (y_train == 0).sum()
    class_weights = [1, num_neg / num_pos]

    model = CatBoostClassifier(
        iterations=n_estimators,
        random_seed=42,
        class_weights=class_weights,
        verbose=0  # 关闭训练输出
    )
    model.fit(X_train, y_train.ravel())
    return model
