from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from autofit.aggregator.aggregator import Aggregator
import pandas as pd

from aggregator_agent.outliers.pca_mahalanobis import pca_mahalanobis_outliers, plot_outlier_diagnostics

parser = ArgumentParser()
parser.add_argument(
    "path",
    type=Path,
    help="Path to the directory containing fits.",
)

args = parser.parse_args()
data = defaultdict(list)

for output in Aggregator.from_directory(
        directory=args.path,
):
    for key, value in output.samples_summary.max_log_likelihood_sample.kwargs.items():
        data[key].append(value)

df = pd.DataFrame(data)

result = pca_mahalanobis_outliers(df, pca_variance=0.95, alpha=0.99, robust=True)
print("Outlier index values:", result.outlier_indices)
print("Outlier integer positions:", result.outlier_positions)
plot_outlier_diagnostics(df, result, pca_variance=0.95)
