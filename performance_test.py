from pathlib import Path
from csv import DictReader
from aggregator_agent.schema import LensFitAnalysis

directory = Path(__file__).parent
data_directory = directory / "data"


class GroundTruth(LensFitAnalysis):
    """
    A ground truth comprising a lens fit analysis and an identifier
    """

    id: str


with open(directory / "image_analysis.csv") as f:
    ground_truths = [GroundTruth.model_validate(row) for row in DictReader(f)]

for ground_truth in ground_truths:
    print(ground_truth)
