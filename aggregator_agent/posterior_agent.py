from io import BytesIO

from PIL import Image
from autofit.aggregator.search_output import AbstractSearchOutput
from pydantic import BaseModel
from pydantic_ai import Agent, BinaryContent

MAX_SIZE = 512
CORNER_PLOT_SYSTEM_PROMPT = """
I will show you a corner plot produced from a Bayesian fit to imaging data.

Your task is to critically assess the quality of the inferred posterior distributions and the overall adequacy of the fit, based solely on the information visible in the corner plot.

In your assessment, consider common indicators of problematic inference, including but not limited to:
1. Posterior distributions that are truncated, clipped, or piled up against the boundaries of their priors, suggesting prior domination or parameter non-identifiability.
2. Posteriors that remain broad, flat, or strongly prior-like, indicating that the data do not meaningfully constrain the corresponding parameters.
3. Evidence of multi-modality in one- or two-dimensional marginals (e.g. multiple peaks, separated clusters, or disconnected regions of high probability).
4. Strong parameter degeneracies, nonlinear correlations, or ridge-like structures that may undermine parameter interpretability.
5. Asymmetries, heavy tails, or irregular posterior shapes that could indicate model misspecification, insufficient data, or numerical/sampling issues.
6. Inconsistencies between marginal and joint distributions that suggest unstable or poorly explored posterior structure.

If there is any indication of multi-modal structure, explicitly state this and identify the affected parameters where possible.

You may also comment on any other posterior pathologies or anomalies not explicitly listed above if they are evident from the plot.

Conclude with a concise overall judgement of fit quality (e.g. well-constrained, weakly constrained, or problematic) and briefly justify your assessment.
"""


class Result(BaseModel):
    """
    The result of the posterior fit analysis.

    Attributes:
        explanation (str): A concise explanation of the fit quality.
        is_good_fit (bool): Whether the fit is considered good or not.
        may_be_multi_modal (bool): Whether the posterior distributions may be multi-modal.
    """
    explanation: str
    is_good_fit: bool
    may_be_multi_modal: bool


class PosteriorFitAnalysis:
    def __init__(
            self,
            search_output: AbstractSearchOutput,
            max_image_size: int = MAX_SIZE,
    ):
        """
        Analyzes the posterior fit quality using a corner plot and a VLM.

        Parameters
        ----------
        search_output : AbstractSearchOutput
            The search output containing the corner plot image.
        max_image_size : int
            The maximum size (in pixels) for the longest side of the image.
        """
        self.search_output = search_output
        self.max_image_size = max_image_size

    @property
    def _image_bytes(self) -> bytes:
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
        """
        Analyzes the corner plot image using a VLM to assess fit quality.
        """
        agent = Agent(
            model="gpt-5.2",
            instructions=CORNER_PLOT_SYSTEM_PROMPT,
            output_type=Result,
        )

        return agent.run_sync(
            [
                BinaryContent(
                    data=self._image_bytes,
                    media_type="image/png",
                ),
            ]
        ).output
