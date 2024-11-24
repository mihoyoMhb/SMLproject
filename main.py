from sklearn.model_selection import train_test_split
from scripts.data_preprocessing import load_data, preprocess_data
from scripts.correlation_analysis import calculate_spearman_correlation, plot_correlation_matrix
from scripts.scaling_pca import scale_data, apply_pca
from scripts.class_balancing import balance_classes
from scripts.model_training import train_logistic_regression, train_random_forest, train_xgboost, train_lightgbm, \
    train_catboost
from scripts.evaluation import evaluate_model
from scripts.hyperparameter_tuning import tune_random_forest, tune_xgboost, tune_catboost
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


# Load and preprocess data
data = load_data('data/training_data_fall2024.csv')
data = preprocess_data(data)

# Correlation Analysis
numeric_data = data.select_dtypes(include=['float64', 'int64'])
spearman_corr, p_values, high_corr_df = calculate_spearman_correlation(numeric_data, threshold=0.7)
plot_correlation_matrix(spearman_corr)

# Print highly correlated pairs
print(f"high_corr_df = \n{high_corr_df}")

# Scaling and PCA
scaled_data = scale_data(numeric_data)
pca_data = apply_pca(scaled_data, n_components=10)

# Split dataset
mapping = {'high_bike_demand': 1, 'low_bike_demand': 0}
y = data['increase_stock'].replace(mapping).to_numpy().reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(pca_data, y, test_size=0.2, random_state=42,
                                                    stratify=y)
# pca_data, y = balance_classes(pca_data, y)


X_train_bal, y_train_bal = balance_classes(X_train, y_train)

# X_train_bal, y_train_bal = X_train, y_train
# Hyperparameter tuning for Random Forest
print("Tuning Random Forest...")
best_rf = tune_random_forest(X_train_bal, y_train_bal)

# Evaluate the tuned Random Forest model
accuracy, f1, report = evaluate_model(best_rf, X_test, y_test)
print(f"Tuned Random Forest Accuracy: {accuracy:.2f}")
print(f"Tuned Random Forest F1 Score: {f1:.2f}")
print("Classification Report:\n", report)
print("Confusion Matrix:\n", confusion_matrix(y_test, best_rf.predict(X_test)))

# Hyperparameter tuning for XGBoost
print("Tuning XGBoost...")
best_xgb = tune_xgboost(X_train_bal, y_train_bal)

# Evaluate the tuned XGBoost model
accuracy, f1, report = evaluate_model(best_xgb, X_test, y_test)
print(f"Tuned XGBoost Accuracy: {accuracy:.2f}")
print(f"Tuned XGBoost F1 Score: {f1:.2f}")
print("Classification Report:\n", report)
print("Confusion Matrix:\n", confusion_matrix(y_test, best_xgb.predict(X_test)))


# # Hyperparameter tuning for CatBoost
# print("Tuning CatBoost...")
# best_cat = tune_catboost(X_train_bal, y_train_bal)
#
# # Evaluate the tuned CatBoost model
# accuracy, f1, report = evaluate_model(best_cat, X_test, y_test)
# print(f"Tuned CatBoost Accuracy: {accuracy:.2f}")
# print(f"Tuned CatBoost F1 Score: {f1:.2f}")
# print("Classification Report:\n", report)
# print("Confusion Matrix:\n", confusion_matrix(y_test, best_cat.predict(X_test)))