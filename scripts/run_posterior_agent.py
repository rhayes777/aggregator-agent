from argparse import ArgumentParser
from pathlib import Path
from autofit.aggregator.aggregator import Aggregator
import csv

from aggregator_agent.posterior.posterior_agent import MAX_SIZE, PosteriorFitAnalysis

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

fieldnames = [
    # existing
    "path",
    "is_good_fit",
    "may_be_multi_modal",

    # richer diagnostics
    "fit_quality",
    "model_complexity",
    "overall_confidence",
    "summary",
    "num_issues",
    "issue_types",
    "severities",
    "recommended_actions",
]

with open(output_file, mode="w", newline="") as csvfile:
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

        # ---- flatten structured issues safely ----
        issues = result.key_issues or []

        issue_types = sorted({issue.issue_type for issue in issues})
        severities = sorted({issue.severity for issue in issues})

        recommended_actions = sorted({
            action
            for issue in issues
            for action in (issue.recommended_actions or [])
        })

        writer.writerow({
            "path": str(output.directory.relative_to(args.path)),
            "is_good_fit": result.is_good_fit,
            "may_be_multi_modal": result.may_be_multi_modal,
            "fit_quality": result.fit_quality,
            "model_complexity": result.model_complexity,
            "overall_confidence": result.overall_confidence,
            "summary": result.summary,
            "num_issues": len(issues),
            "issue_types": "|".join(issue_types),
            "severities": "|".join(severities),
            "recommended_actions": "|".join(recommended_actions),
        })
