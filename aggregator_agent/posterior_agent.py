from io import BytesIO

from PIL import Image
from autofit.aggregator.search_output import AbstractSearchOutput
from pydantic import BaseModel
from pydantic_ai import Agent, BinaryContent

MAX_SIZE = 512
CORNER_PLOT_SYSTEM_PROMPT = """
I will show you the corner plot resultant from a bayesian fit to imaging data.

Describe the quality of the fit.

Reasons the fit may be bad include:
Any posterior distribution appears to be pushed up against the edge of the prior.
The data is unable to constrain the posterior distributions, the fit is bad.
"""


class Result(BaseModel):
    """
    The result of the posterior fit analysis.

    Attributes:
        explanation (str): A concise explanation of the fit quality.
        is_good_fit (bool): Whether the fit is considered good or not.
    """
    explanation: str
    is_good_fit: bool


class PosteriorFitAnalysis:
    def __init__(
            self,
            search_output: AbstractSearchOutput,
            max_image_size: int = MAX_SIZE,
    ):
        self.search_output = search_output
        self.max_image_size = max_image_size

    @property
    def image_bytes(self) -> bytes:
        img = self.search_output.image("search/corner_anesthetic")
        img = img.convert("RGB")
        img.thumbnail(
            (self.max_image_size, self.max_image_size),
            Image.Resampling.LANCZOS,
        )

        buffer = BytesIO()
        img.save(buffer, format="PNG")
        return buffer.getvalue()

    def corner_plot_analysis(self) -> Result:
        agent = Agent(
            model="gpt-5.2",
            instructions=CORNER_PLOT_SYSTEM_PROMPT,
            output_type=Result,
        )

        result = agent.run_sync(
            [
                BinaryContent(
                    data=self.image_bytes,
                    media_type="image/png",
                ),
            ]
        ).output

        return result
