from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from autofit.aggregator.aggregator import Aggregator

from aggregator_agent.outliers.outlier_detection import identify_outliers

parser = ArgumentParser()
parser.add_argument(
    "path",
    type=Path,
    help="Path to the directory containing fits.",
)

args = parser.parse_args()
data = defaultdict(list)

aggregator = Aggregator.from_directory(
    directory=args.path,
)

outliers = identify_outliers(aggregator)

for outlier in outliers:
    print(outlier)
