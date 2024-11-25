# scripts/scaling_pca.py
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np


def scale_data(data):
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)


def apply_pca(data, ratio=0.95):
    pca = PCA()
    pca_components = pca.fit_transform(data)
    pca_df = pd.DataFrame(data=pca_components, index=data.index,
                          columns=[f"PC{i + 1}" for i in range(pca.n_components_)])

    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    n_components = np.argmax(cumulative_variance >= ratio) + 1

    return pca_df.iloc[:, :n_components]
