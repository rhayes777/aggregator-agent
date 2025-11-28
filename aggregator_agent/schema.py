from enum import StrEnum

from pydantic import BaseModel


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


class LensFitAnalysis(BaseModel):
    """
    An analysis of images output from a lensing fit.

    Attributes
    ---------
    category - the category to which this image belongs
    description - a brief description of the image highlighting why it belongs to the given category
    """

    category: Category
    description: str
