from pathlib import Path
from csv import DictReader

from aggregator_agent.image_agent import categorise
from aggregator_agent.schema import LensFitAnalysis

directory = Path(__file__).parent
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

ground_truth = ground_truths[0]

print("Expected")
print(ground_truth.model_dump_json(indent=2))

print("\nActual")
print(categorise(ground_truth.image_path).model_dump_json(indent=2))
