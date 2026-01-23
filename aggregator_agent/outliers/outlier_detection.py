from collections import defaultdict

from autofit import SearchOutput
import pandas as pd
from autofit.database.aggregator.aggregator import AbstractAggregator, Aggregator

from aggregator_agent.outliers.pca_mahalanobis import (
    pca_mahalanobis_outliers,
)
import logging


logger = logging.getLogger(__name__)


def identify_outliers(
    aggregator: AbstractAggregator | Aggregator,
    pca_variance=0.95,
    alpha=0.99,
    robust=True,
) -> list[SearchOutput]:
    """
    Identify outlier search outputs in the given aggregator using PCA and Mahalanobis distance.

    Parameters
    ----------
    aggregator
        The aggregator containing search outputs to analyze.
    pca_variance
        The amount of variance to retain in PCA.
    alpha
        The significance level for outlier detection. Higher values result in fewer outliers.
    robust
        Whether to use robust covariance estimation.

    Returns
    -------
    A list of search outputs identified as outliers.
    """
    data = defaultdict(list)
    search_outputs = list(aggregator)

    for output in search_outputs:
        try:
            for (
                key,
                value,
            ) in output.samples_summary.max_log_likelihood_sample.kwargs.items():
                data[key].append(value)
        except Exception as e:
            logger.exception(f"Failed to process output {output}: {e}")

    df = pd.DataFrame(data)

    result = pca_mahalanobis_outliers(
        df,
        pca_variance=pca_variance,
        alpha=alpha,
        robust=robust,
    )

    return [search_outputs[index] for index in result.outlier_indices]
