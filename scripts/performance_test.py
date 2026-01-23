import csv
import random
from pathlib import Path
from csv import DictReader
import datetime as dt

from aggregator_agent.image_agent import categorise
from aggregator_agent.schema import LensFitAnalysis, Category

directory = Path(__file__).parents[1]
data_directory = directory / "data"
initial_lens_model_directory = data_directory / "initial_lens_model"


class GroundTruth(LensFitAnalysis):
    """
    A ground truth comprising a lens fit analysis and an identifier
    """

    id: str

    @property
    def image_path(self) -> Path:
        """
        The path to the test data. Assumed to be in data/initial_lens_model.
        """
        return (initial_lens_model_directory / self.id).with_suffix(".png")


with open(directory / "image_analysis.csv") as f:
    ground_truths = [GroundTruth.model_validate(row) for row in DictReader(f)]

ground_truth_paths = {ground_truth.image_path for ground_truth in ground_truths}

good_ground_truths = []

for file in initial_lens_model_directory.iterdir():
    if file not in ground_truth_paths:
        good_ground_truths.append(
            GroundTruth(
                id=file.stem,
                category=Category.Good,
                description="Good",
            )
        )

random.shuffle(good_ground_truths)
ground_truths.extend(good_ground_truths[: len(ground_truths)])

random.shuffle(ground_truths)

timestamp = dt.datetime.now().isoformat()

with open(f"results-{timestamp}.csv", "w+") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            "id",
            "expected_category",
            "predicted_category",
            "expected_description",
            "predicted_description",
        ]
    )
    for ground_truth in ground_truths:
        predicted = categorise(ground_truth.image_path)
        print(f"Expected: {ground_truth} ; Predicted {predicted}")
        writer.writerow(
            [
                ground_truth.id,
                ground_truth.category,
                predicted.category,
                ground_truth.description,
                predicted.description,
            ]
        )
