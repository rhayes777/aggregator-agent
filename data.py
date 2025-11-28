from enum import StrEnum

from pathlib import Path
from csv import DictReader
from pydantic import BaseModel

directory = Path(__file__).parent
data_directory = directory / "data"


class Category(StrEnum):
    """
    Categories for image analysis results.

    - BadModelIsLens: the lens system is real, but the modelling setup is inadequate
    - GoodModelNotLens: These systems are not gravitational lenses, and the good model correctly shows no convincing
        counter image
    - MightBeLensBadModel: These systems show uncertain or ambiguous evidence for lensing, and the model cannot reliably
        confirm or rule out a lens
    - DataIssue: Systems in this category suffer from data-quality problems that prevent meaningful modelling or
        interpretation
    - Fixable: These systems are not fundamentally bad models or bad data—they simply require improved masking to work
        correctly.
    - Good: the model fits well and the classification is reliable
    """

    BadModelIsLens = "BadModelIsLens"
    GoodModelNotLens = "GoodModelNotLens"
    MightBeLensBadModel = "MightBeLensBadModel"
    DataIssue = "DataIssue"
    Fixable = "Fixable"
    Good = "Good"


class ImageAnalysis(BaseModel):
    category: Category
    id: str
    description: str


with open(directory / "image_analysis.csv") as f:
    categories = {row["Category"] for row in DictReader(f)}

for category in categories:
    print(category)
