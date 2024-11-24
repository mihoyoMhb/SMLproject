# scripts/normality_tests.py
import pandas as pd
from scipy.stats import shapiro, normaltest, kstest, probplot
import matplotlib.pyplot as plt


def normality_analysis(data, column_name):
    # Perform statistical tests
    stat_shapiro, p_shapiro = shapiro(data[column_name])
    stat_dagostino, p_dagostino = normaltest(data[column_name])
    stat_ks, p_ks = kstest(data[column_name], 'norm')

    # Create a summary DataFrame
    results = pd.DataFrame({
        "Test": ["Shapiro-Wilk", "D'Agostino and Pearson", "Kolmogorov-Smirnov"],
        "Statistic": [stat_shapiro, stat_dagostino, stat_ks],
        "P-Value": [p_shapiro, p_dagostino, p_ks]
    })
    print(f"Normality Test Results for {column_name}")
    print(results)

    # Create histograms and Q-Q plots
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.hist(data[column_name], bins=30, color='blue', edgecolor='black', alpha=0.7)
    plt.title(f"Histogram of {column_name}")
    plt.xlabel(column_name)
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    probplot(data[column_name], dist="norm", plot=plt)
    plt.title(f"Q-Q Plot of {column_name}")
    plt.tight_layout()
    plt.show()
