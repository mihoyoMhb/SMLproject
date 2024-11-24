# scripts/correlation_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


def calculate_spearman_correlation(data, threshold=0.7):
    """
    Calculate the Spearman correlation matrix and identify highly correlated pairs.

    Args:
        data (DataFrame): The input dataframe containing the features.
        threshold (float): The correlation threshold to consider features as highly correlated.

    Returns:
        spearman_correlation_matrix (DataFrame): The Spearman correlation matrix.
        p_value_matrix (DataFrame): The p-value matrix corresponding to the correlations.
        high_corr_df (DataFrame): DataFrame of highly correlated feature pairs.
    """
    # Calculate Spearman correlation and p-value matrices
    columns = data.columns
    spearman_matrix = pd.DataFrame(index=columns, columns=columns)
    p_value_matrix = pd.DataFrame(index=columns, columns=columns)

    for col1 in columns:
        for col2 in columns:
            if col1 != col2:
                corr, p_value = spearmanr(data[col1], data[col2])
                spearman_matrix.loc[col1, col2] = corr
                p_value_matrix.loc[col1, col2] = p_value
            else:
                spearman_matrix.loc[col1, col2] = 1
                p_value_matrix.loc[col1, col2] = 0

    spearman_matrix = spearman_matrix.astype(float)
    p_value_matrix = p_value_matrix.astype(float)

    # Identify highly correlated pairs based on threshold and p-value < 0.05
    high_corr_pairs = [
        (spearman_matrix.index[i], spearman_matrix.columns[j], spearman_matrix.iloc[i, j])
        for i in range(len(spearman_matrix.columns))
        for j in range(i + 1, len(spearman_matrix.columns))
        if abs(spearman_matrix.iloc[i, j]) > threshold and p_value_matrix.iloc[i, j] < 0.05
    ]

    high_corr_df = pd.DataFrame(high_corr_pairs, columns=["Feature 1", "Feature 2", "Correlation"])
    # print(f"Highly Correlated Pairs (Threshold > {threshold}):\n{high_corr_df}")

    return spearman_matrix, p_value_matrix, high_corr_df


def plot_correlation_matrix(matrix, title='Spearman Correlation Matrix Heatmap'):
    """
    Plot the Spearman correlation matrix as a heatmap.

    Args:
        matrix (DataFrame): The correlation matrix to be plotted.
        title (str): The title of the heatmap.
    """
    plt.figure(figsize=(12, 8))
    plt.imshow(matrix, cmap='coolwarm', interpolation='none')
    plt.colorbar(label='Spearman Correlation Coefficient')
    plt.xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha='right')
    plt.yticks(range(len(matrix.columns)), matrix.columns)
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.show()
