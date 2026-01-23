"""
Uses a VLM to analyse posterior distributions by inspecting corner plots.
"""

from argparse import ArgumentParser
from pathlib import Path
from autofit.aggregator.aggregator import Aggregator
import csv

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

output_file = args.path / "posterior_analysis_results.csv"

with open(output_file, mode="w", newline="") as csvfile:
    fieldnames = ["path", "explanation", "is_good_fit", "may_be_multi_modal"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

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

        writer.writerow(
            {
                "path": str(output.directory.relative_to(args.path)),
                "explanation": result.explanation,
                "is_good_fit": result.is_good_fit,
                "may_be_multi_modal": result.may_be_multi_modal,
            }
        )
