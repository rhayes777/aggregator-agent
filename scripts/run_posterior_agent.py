from argparse import ArgumentParser
from pathlib import Path

from aggregator_agent.posterior_agent import MAX_SIZE, PosteriorFitAnalysis

parser = ArgumentParser()
parser.add_argument(
    "path",
    type=Path,
    help="Path to the fit directory.",
)
parser.add_argument(
    "--max-size",
    type=int,
    default=MAX_SIZE,
    help=f"Maximum size (in pixels) for the longest side of the image. Default is {MAX_SIZE}.",
)
args = parser.parse_args()

analysis = PosteriorFitAnalysis(
    fit_path=args.path,
    max_image_size=args.max_size,
)

result = analysis.corner_plot_analysis()
print("Corner Plot Analysis Result:")
print(result)
