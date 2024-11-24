# scripts/scaling_pca.py
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np


def scale_data(data):
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)


def apply_pca(data, n_components=9):
    pca = PCA()
    pca_components = pca.fit_transform(data)
    pca_df = pd.DataFrame(data=pca_components, index=data.index,
                          columns=[f"PC{i + 1}" for i in range(pca.n_components_)])
    return pca_df.iloc[:, :n_components]
