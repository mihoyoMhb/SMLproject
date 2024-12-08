import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier



def label_weather_cluster(cluster_label):
    if cluster_label == 1:
        return 'bad_weather'
    elif cluster_label == 0:
        return 'good_weather'



class CustomPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=3, random_state=42)
        self.weather_dummies_columns = None
        self.categorical_features = None
        self.numeric_features = None

    def fit(self, X, y=None):
        """
        :param X:
        :param y:
        :return: self
        """
        # 对数变换
        X = X.copy()
        X['visibility'] = np.log1p(X['visibility'])
        X['snowdepth'] = np.log1p(X['snowdepth'])
        X['precip'] = np.log1p(X['precip'])
        # 时间特征转换
        X['month_sin'] = np.sin(2 * np.pi * X['month'] / 12)
        X['month_cos'] = np.cos(2 * np.pi * X['month'] / 12)
        X['day_of_week_sin'] = np.sin(2 * np.pi * X['day_of_week'] / 7)
        X['day_of_week_cos'] = np.cos(2 * np.pi * X['day_of_week'] / 7)
        X['hour_of_day_sin'] = np.sin(2 * np.pi * X['hour_of_day'] / 24)
        X['hour_of_day_cos'] = np.cos(2 * np.pi * X['hour_of_day'] / 24)

        X = X.drop(columns=['month', 'day_of_week', 'hour_of_day', 'weekday', 'temp', 'snow'], axis=1)

        clustering_features = ['dew', 'humidity', 'snowdepth', 'windspeed', 'cloudcover',
                               'visibility', 'precip']
        # 在训练数据上拟合 KMeans
        self.kmeans.fit(X[clustering_features])
        # 添加聚类结果
        X['weather_cluster'] = self.kmeans.labels_
        # 映射聚类标签到天气质量
        X['weather_quality'] = X['weather_cluster'].apply(label_weather_cluster)
        # One-Hot 编码
        weather_dummies = pd.get_dummies(X['weather_quality'], prefix='whether', drop_first=True)
        # 保存天气哑变量的列名
        self.weather_dummies_columns = weather_dummies.columns
        X = pd.concat([X, weather_dummies], axis=1)
        X = X.drop(columns=clustering_features, axis=1)
        X = X.drop(columns=['weather_quality', 'weather_cluster'], axis=1)
        # 选择特征
        weather_features = []
        self.categorical_features = ['holiday'] + list(self.weather_dummies_columns)
        self.numeric_features = [col for col in X.columns if
                                 col not in ['holiday', 'weekday', 'increase_stock', 'weather_cluster',
                                             'weather_quality'] + list(self.weather_dummies_columns) + weather_features]
        # 转换布尔类型
        for col in X.select_dtypes(include=['bool']).columns:
            X[col] = X[col].astype(int)
        # 在训练数据上拟合 StandardScaler
        # print(X)
        self.scaler.fit(X[self.numeric_features])
        return self

    def transform(self, X):
        """这个函数看似重复，但是不能动"""
        X = X.copy()

        X['visibility'] = np.log1p(X['visibility'])
        X['snowdepth'] = np.log1p(X['snowdepth'])
        X['precip'] = np.log1p(X['precip'])

        X['month_sin'] = np.sin(2 * np.pi * X['month'] / 12)
        X['month_cos'] = np.cos(2 * np.pi * X['month'] / 12)
        X['day_of_week_sin'] = np.sin(2 * np.pi * X['day_of_week'] / 7)
        X['day_of_week_cos'] = np.cos(2 * np.pi * X['day_of_week'] / 7)
        X['hour_of_day_sin'] = np.sin(2 * np.pi * X['hour_of_day'] / 24)
        X['hour_of_day_cos'] = np.cos(2 * np.pi * X['hour_of_day'] / 24)

        X = X.drop(columns=['month', 'day_of_week', 'hour_of_day', 'weekday', 'temp', 'snow'], axis=1)

        clustering_features = ['dew', 'humidity', 'snowdepth', 'windspeed', 'cloudcover',
                               'visibility', 'precip']

        X['weather_cluster'] = self.kmeans.predict(X[clustering_features])

        X['weather_quality'] = X['weather_cluster'].apply(label_weather_cluster)

        weather_dummies = pd.get_dummies(X['weather_quality'], prefix='whether', drop_first=True)

        for col in self.weather_dummies_columns:
            if col not in weather_dummies.columns:
                weather_dummies[col] = 0
        X = pd.concat([X, weather_dummies], axis=1)
        X = X.drop(columns=clustering_features, axis=1)
        for col in X.select_dtypes(include=['bool']).columns:
            X[col] = X[col].astype(int)

        X_scaled = self.scaler.transform(X[self.numeric_features])
        X_scaled = pd.DataFrame(X_scaled, columns=self.numeric_features, index=X.index)

        X_processed = pd.concat([X[self.categorical_features], X_scaled], axis=1)
        # print(X_processed)
        return X_processed


def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=['low_bike_demand', 'high_bike_demand'])
    return accuracy, f1, report


def tune_random_forest_rs(X_train, y_train, cv=10, scoring='f1', n_iter=200):
    num_pos = np.sum(y_train == 1)
    num_neg = np.sum(y_train == 0)
    ratio = num_neg / num_pos
    param_dist = {
        'classifier__n_estimators': [2 * i for i in range(250, 300)],
        'classifier__max_depth': list(range(15, 25)),
        'classifier__min_samples_split': list(range(3, 32)),
        'classifier__min_samples_leaf': list(range(3, 32)),
        'classifier__max_features': ['sqrt', 'log2', 0.1, 0.2, 0.3, 0.4, 0.5],
        'classifier__bootstrap': [True, False],
        'classifier__class_weight': ['balanced', {0: 1, 1: ratio}, {0: 1 / ratio, 1: 1}, {0: 1, 1: 1}],
        'classifier__criterion': ["gini", "entropy", "log_loss"],
        'classifier__warm_start': [False],
    }
    # 创建 Pipeline
    """这里也是最关键的步骤，需要这里加入pipeline"""
    pipeline = Pipeline([
        ('preprocessor', CustomPreprocessor()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        random_state=42
    )
    random_search.fit(X_train, y_train.ravel())
    return random_search.best_estimator_


def tune_ada_boost(X_train, y_train, cv=10, scoring='f1', n_iter=10):
    param_dist = {
        'classifier__n_estimators': [10 * i for i in range(10, 101)],
        'classifier__learning_rate': [0.01, 0.1, 0.2, 0.3, 0,4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        'classifier__algorithm': ['SAMME']
    }
    pipeline = Pipeline([
        ('preprocessor', CustomPreprocessor()),
        ('classifier', AdaBoostClassifier(random_state=42))
    ])
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        random_state=42
    )
    random_search.fit(X_train, y_train.ravel())
    return random_search.best_estimator_


"""
    LDA和QDA有点奇怪，但是先这样写吧
"""
def train_lda(X_train, y_train):
    param_dist = {
        'classifier__solver': ['lsqr', 'eigen'],
        'classifier__shrinkage': [None, 'auto', 0.1, 0.5],
    }
    pipeline = Pipeline([
        ('preprocessor', CustomPreprocessor()),
        ('classifier', LinearDiscriminantAnalysis())
    ])
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=10,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        random_state=42
    )
    random_search.fit(X_train, y_train.ravel())
    return random_search.best_estimator_


def train_qda(X_train, y_train):
    # 对QDA进行超参数搜索
    param_dist = {
        'classifier__reg_param': [0.0, 0.1, 0.3, 0.5, 0.7, 0.9],
        'classifier__store_covariance': [True, False],
        'classifier__tol': [1e-4, 1e-3, 1e-2, 1e-1]
    }
    pipeline = Pipeline([
        ('preprocessor', CustomPreprocessor()),
        ('classifier', QuadraticDiscriminantAnalysis())
    ])
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=10,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        random_state=42
    )
    random_search.fit(X_train, y_train.ravel())
    return random_search.best_estimator_

def tune_logistic_regression(X_train, y_train, cv=10, scoring='f1', n_iter=10):
    num_pos = np.sum(y_train == 1)
    num_neg = np.sum(y_train == 0)
    ratio = num_neg / num_pos
    param_dist = {
        'classifier__C': np.logspace(-3, 3, 7),
        'classifier__penalty': ['l2'],
        'classifier__class_weight': ['balanced', {0: 1, 1: ratio}],
        'classifier__solver': ['lbfgs', 'liblinear'],
        'classifier__max_iter': [100, 500, 1000],
    }
    pipeline = Pipeline([
        ('preprocessor', CustomPreprocessor()),
        ('classifier', LogisticRegression(random_state=42))
    ])
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        random_state=42
    )
    random_search.fit(X_train, y_train.ravel())
    return random_search.best_estimator_


def tune_knn(X_train, y_train, cv=10, scoring='f1', n_iter=10):
    param_dist = {
        'classifier__n_neighbors': list(range(1, 31)),
        'classifier__weights': ['uniform', 'distance'],
        'classifier__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'classifier__p': [1, 2],  # p=1 曼哈顿距离，p=2 欧氏距离
    }
    pipeline = Pipeline([
        ('preprocessor', CustomPreprocessor()),
        ('classifier', KNeighborsClassifier())
    ])
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    random_search.fit(X_train, y_train.ravel())
    return random_search.best_estimator_


# 主程序
if __name__ == "__main__":
    # 加载数据, 文件地址自己改一下
    data = pd.read_csv('data/training_data_fall2024.csv')
    # models
    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'LDA': LinearDiscriminantAnalysis(),
        'QDA': QuadraticDiscriminantAnalysis(),
        'AdaBoost': AdaBoostClassifier(random_state=42)
    }
    # 定义特征和目标变量
    X_all = data.copy()
    y_all = data['increase_stock'].map({'low_bike_demand': 0, 'high_bike_demand': 1}).to_numpy().ravel()
    X_all = X_all.drop(columns=['increase_stock'])
    X_trainval, X_test, y_trainval, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42,
                                                              stratify=y_all)
    # 存储结果
    results = {}

    # 训练 Random Forest
    print("\nTraining Random Forest...")
    best_rf = tune_random_forest_rs(X_trainval, y_trainval)
    y_pred_rf = best_rf.predict(X_test)
    accuracy_rf, f1_rf, report_rf = evaluate_model(y_test, y_pred_rf)
    print(f"Random Forest Test Set Accuracy: {accuracy_rf:.2f}")
    print(f"Random Forest Test Set F1 Score: {f1_rf:.2f}")
    print(f"Random Forest Classification Report:\n{report_rf}")
    print(f"Random Forest Confusion Matrix:\n{confusion_matrix(y_test, y_pred_rf)}")
    results['Random Forest'] = {
        'model': best_rf,
        'accuracy': accuracy_rf,
        'f1': f1_rf,
        'report': report_rf,
        'confusion_matrix': confusion_matrix(y_test, y_pred_rf)
    }

    # 训练 Logistic Regression
    print("\nTraining Logistic Regression...")
    best_lr = tune_logistic_regression(X_trainval, y_trainval)
    y_pred_lr = best_lr.predict(X_test)
    accuracy_lr, f1_lr, report_lr = evaluate_model(y_test, y_pred_lr)
    print(f"Logistic Regression Test Set Accuracy: {accuracy_lr:.2f}")
    print(f"Logistic Regression Test Set F1 Score: {f1_lr:.2f}")
    print(f"Logistic Regression Classification Report:\n{report_lr}")
    print(f"Logistic Regression Confusion Matrix:\n{confusion_matrix(y_test, y_pred_lr)}")
    results['Logistic Regression'] = {
        'model': best_lr,
        'accuracy': accuracy_lr,
        'f1': f1_lr,
        'report': report_lr,
        'confusion_matrix': confusion_matrix(y_test, y_pred_lr)
    }

    # 训练 AdaBoost
    print("\nTraining AdaBoost...")
    best_ada = tune_ada_boost(X_trainval, y_trainval)
    y_pred_ada = best_ada.predict(X_test)
    accuracy_ada, f1_ada, report_ada = evaluate_model(y_test, y_pred_ada)
    print(f"AdaBoost Test Set Accuracy: {accuracy_ada:.2f}")
    print(f"AdaBoost Test Set F1 Score: {f1_ada:.2f}")
    print(f"AdaBoost Classification Report:\n{report_ada}")
    print(f"AdaBoost Confusion Matrix:\n{confusion_matrix(y_test, y_pred_ada)}")
    results['AdaBoost'] = {
        'model': best_ada,
        'accuracy': accuracy_ada,
        'f1': f1_ada,
        'report': report_ada,
        'confusion_matrix': confusion_matrix(y_test, y_pred_ada)
    }

    # 训练 LDA
    print("\nTraining LDA...")
    best_lda = train_lda(X_trainval, y_trainval)
    y_pred_lda = best_lda.predict(X_test)
    accuracy_lda, f1_lda, report_lda = evaluate_model(y_test, y_pred_lda)
    print(f"LDA Test Set Accuracy: {accuracy_lda:.2f}")
    print(f"LDA Test Set F1 Score: {f1_lda:.2f}")
    print(f"LDA Classification Report:\n{report_lda}")
    print(f"LDA Confusion Matrix:\n{confusion_matrix(y_test, y_pred_lda)}")
    results['LDA'] = {
        'model': best_lda,
        'accuracy': accuracy_lda,
        'f1': f1_lda,
        'report': report_lda,
        'confusion_matrix': confusion_matrix(y_test, y_pred_lda)
    }

    # 训练 QDA
    print("\nTraining QDA...")
    best_qda = train_qda(X_trainval, y_trainval)
    y_pred_qda = best_qda.predict(X_test)
    accuracy_qda, f1_qda, report_qda = evaluate_model(y_test, y_pred_qda)
    print(f"QDA Test Set Accuracy: {accuracy_qda:.2f}")
    print(f"QDA Test Set F1 Score: {f1_qda:.2f}")
    print(f"QDA Classification Report:\n{report_qda}")
    print(f"QDA Confusion Matrix:\n{confusion_matrix(y_test, y_pred_qda)}")
    results['QDA'] = {
        'model': best_qda,
        'accuracy': accuracy_qda,
        'f1': f1_qda,
        'report': report_qda,
        'confusion_matrix': confusion_matrix(y_test, y_pred_qda)
    }

    print("\nTraining KNN...")
    best_knn = tune_knn(X_trainval, y_trainval)
    y_pred_knn = best_knn.predict(X_test)
    accuracy_knn, f1_knn, report_knn = evaluate_model(y_test, y_pred_knn)
    print(f"KNN Test Set Accuracy: {accuracy_knn:.2f}")
    print(f"KNN Test Set F1 Score: {f1_knn:.2f}")
    print(f"KNN Classification Report:\n{report_knn}")
    print(f"KNN Confusion Matrix:\n{confusion_matrix(y_test, y_pred_knn)}")
    results['KNN'] = {
        'model': best_knn,
        'accuracy': accuracy_knn,
        'f1': f1_knn,
        'report': report_knn,
        'confusion_matrix': confusion_matrix(y_test, y_pred_knn)
    }

    # 输出所有模型的结果
    print("\nSummary of Model Performance:")
    for model_name, result in results.items():
        print(f"{model_name} Test Set Accuracy: {result['accuracy']:.2f}")
        print(f"{model_name} Test Set F1 Score: {result['f1']:.2f}\n")