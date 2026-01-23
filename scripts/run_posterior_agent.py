from argparse import ArgumentParser
from pathlib import Path
from autofit.aggregator.aggregator import Aggregator

from aggregator_agent.posterior_agent import MAX_SIZE, PosteriorFitAnalysis

parser = ArgumentParser()
parser.add_argument(
    "path",
    type=Path,
    help="Path to the directory containing fits.",
)
parser.add_argument(
    "--max-size",
    type=int,
    default=MAX_SIZE,
    help=f"Maximum size (in pixels) for the longest side of the image. Default is {MAX_SIZE}.",
)
args = parser.parse_args()

for output in Aggregator.from_directory(
    directory=args.path,
):

    analysis = PosteriorFitAnalysis(
        search_output=output,
        max_image_size=args.max_size,
    )

    result = analysis.corner_plot_analysis()
    print("Corner Plot Analysis Result:")
    print(result)
