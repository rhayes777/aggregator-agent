from collections import defaultdict

from autofit import SearchOutput
import pandas as pd
from autofit.database.aggregator.aggregator import AbstractAggregator

from aggregator_agent.outliers.pca_mahalanobis import pca_mahalanobis_outliers, plot_outlier_diagnostics


def identify_outliers(aggregator: AbstractAggregator, ) -> list[SearchOutput]:
    """
    Identify outlier search outputs in the given aggregator using PCA and Mahalanobis distance.

    Parameters
    ----------
    aggregator : AbstractAggregator
        The aggregator containing search outputs to analyze.

    Returns
    -------
    A list of search outputs identified as outliers.
    """
    data = defaultdict(list)
    search_outputs = list(aggregator)

    for output in search_outputs:
        for key, value in output.samples_summary.max_log_likelihood_sample.kwargs.items():
            data[key].append(value)

    df = pd.DataFrame(data)

    result = pca_mahalanobis_outliers(df, pca_variance=0.95, alpha=0.99, robust=True)
    plot_outlier_diagnostics(df, result, pca_variance=0.95, random_state=0, show=True)

    return [
        search_outputs[index] for index in result.outlier_indices
    ]
