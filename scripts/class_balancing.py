# scripts/class_balancing.py
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE


def balance_classes(X, y, need_balancing=True):
    if need_balancing:
        # rus = RandomUnderSampler(random_state=42, replacement=True)
        # X_resampled, y_resampled = rus.fit_resample(X, y)
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
    else:
        X_resampled, y_resampled = X, y
    return X_resampled, y_resampled
