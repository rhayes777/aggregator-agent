from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Sequence

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import MinCovDet

from scipy.stats import chi2


@dataclass(frozen=True)
class OutlierResult:
    outlier_mask: np.ndarray  # shape (n,)
    outlier_indices: np.ndarray  # index values from df.index
    outlier_positions: np.ndarray  # integer positions [0..n-1]
    md: np.ndarray  # robust Mahalanobis distances (sqrt form)
    md2: np.ndarray  # robust Mahalanobis squared distances
    threshold_md2: float  # chi2 threshold in squared-space
    pca_components: int
    explained_variance_ratio: np.ndarray


def pca_mahalanobis_outliers(
    df: pd.DataFrame,
    *,
    cols: Sequence[str] | None = None,
    pca_variance: float = 0.95,
    alpha: float = 0.99,
    robust: bool = True,
    random_state: int = 0,
) -> OutlierResult:
    """
    Detect multivariate outliers via Standardize -> PCA -> (Robust) Mahalanobis.

    Parameters
    ----------
    df:
        Input dataframe.
    cols:
        Columns to use. If None, uses all numeric columns.
    pca_variance:
        Fraction of variance to retain in PCA (e.g. 0.95).
    alpha:
        Chi-square quantile for thresholding (e.g. 0.99).
    robust:
        If True, uses MinCovDet for robust covariance; otherwise uses classical covariance.
    random_state:
        For MinCovDet reproducibility.

    Returns
    -------
    OutlierResult
    """
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not cols:
        raise ValueError("No numeric columns found (or provided).")

    X = df.loc[:, cols].to_numpy(dtype=float)

    # Drop rows with any NaNs in selected columns (track mapping back to df.index)
    valid_mask = np.isfinite(X).all(axis=1)
    if not valid_mask.all():
        df_valid = df.loc[valid_mask, :]
        X = X[valid_mask]
    else:
        df_valid = df

    if X.shape[0] < 5:
        raise ValueError(f"Not enough valid rows after NaN filtering: {X.shape[0]}")

    # 1) Standardize
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # 2) PCA (retain variance)
    pca = PCA(n_components=pca_variance, svd_solver="full", random_state=random_state)
    Xp = pca.fit_transform(Xs)
    k = Xp.shape[1]
    if k < 1:
        raise ValueError("PCA produced 0 components; check input variance/columns.")

    # 3) Mahalanobis (robust or classical) in PCA space
    if robust:
        mcd = MinCovDet(random_state=random_state).fit(Xp)
        md2 = mcd.mahalanobis(Xp)  # squared distances
    else:
        mu = Xp.mean(axis=0)
        cov = np.cov(Xp, rowvar=False)
        cov_inv = np.linalg.pinv(cov)
        d = Xp - mu
        md2 = np.einsum("ij,jk,ik->i", d, cov_inv, d)

    md = np.sqrt(md2)

    # 4) Threshold via chi-square on squared distances
    threshold_md2 = float(chi2.ppf(alpha, df=k))
    outlier_mask_valid = md2 > threshold_md2

    # Map back to original df indices/positions
    outlier_positions_valid = np.flatnonzero(outlier_mask_valid)
    outlier_indices = df_valid.index[outlier_positions_valid].to_numpy()

    # Create full-length mask aligned to original df
    outlier_mask_full = np.zeros(df.shape[0], dtype=bool)
    outlier_mask_full[np.flatnonzero(valid_mask)[outlier_positions_valid]] = True

    outlier_positions_full = np.flatnonzero(outlier_mask_full)

    return OutlierResult(
        outlier_mask=outlier_mask_full,
        outlier_indices=outlier_indices,
        outlier_positions=outlier_positions_full,
        md=md,
        md2=md2,
        threshold_md2=threshold_md2,
        pca_components=k,
        explained_variance_ratio=pca.explained_variance_ratio_,
    )


def plot_outlier_diagnostics(
    df: pd.DataFrame,
    result: OutlierResult,
    *,
    cols: Sequence[str] | None = None,
    pca_variance: float = 0.95,
    random_state: int = 0,
    show: bool = True,
) -> None:
    """
    Creates standard diagnostics:
      1) MD^2 histogram + threshold line
      2) MD^2 vs sample index
      3) PCA scatter (PC1 vs PC2) with outliers highlighted (when >=2 PCs)
    """
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Recompute PCA scores for plotting (keeps this function stateless and simple)
    X = df.loc[:, cols].to_numpy(dtype=float)
    valid_mask = np.isfinite(X).all(axis=1)
    X = X[valid_mask]

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pca = PCA(n_components=pca_variance, svd_solver="full", random_state=random_state)
    Xp = pca.fit_transform(Xs)

    # Align distances with "valid rows" only (result.md2 is on valid rows)
    md2 = result.md2
    thr = result.threshold_md2
    outlier_mask_valid = md2 > thr

    # --- Plot 1: histogram of MD^2 ---
    plt.figure()
    plt.hist(md2, bins=40)
    plt.axvline(thr, linewidth=2)
    plt.title("Robust Mahalanobis Distance Squared (MD^2) with Chi-square Threshold")
    plt.xlabel("MD^2")
    plt.ylabel("Count")

    # --- Plot 2: MD^2 vs sample order (valid rows) ---
    plt.figure()
    x = np.arange(md2.shape[0])
    plt.scatter(x, md2, s=15)
    plt.axhline(thr, linewidth=2)
    plt.title("MD^2 by Row (valid rows)")
    plt.xlabel("Row position (after NaN filtering)")
    plt.ylabel("MD^2")

    # --- Plot 3: PCA scatter (PC1 vs PC2) ---
    if Xp.shape[1] >= 2:
        plt.figure()
        plt.scatter(Xp[~outlier_mask_valid, 0], Xp[~outlier_mask_valid, 1], s=20)
        plt.scatter(Xp[outlier_mask_valid, 0], Xp[outlier_mask_valid, 1], s=30)
        plt.title("PCA Scatter (PC1 vs PC2) with Outliers Highlighted")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
    else:
        plt.figure()
        plt.scatter(np.arange(Xp.shape[0]), Xp[:, 0], s=15)
        plt.scatter(np.flatnonzero(outlier_mask_valid), Xp[outlier_mask_valid, 0], s=30)
        plt.title("PCA Scores (PC1) with Outliers Highlighted")
        plt.xlabel("Row position (after NaN filtering)")
        plt.ylabel("PC1 score")

    if show:
        plt.show()
