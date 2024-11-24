# scripts/evaluation.py
from sklearn.metrics import accuracy_score, classification_report, f1_score


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['low_bike_demand', 'high_bike_demand'])
    return accuracy, f1, report
