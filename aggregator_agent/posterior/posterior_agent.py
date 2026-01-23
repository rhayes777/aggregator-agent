from io import BytesIO

from PIL import Image
from autofit.aggregator.search_output import AbstractSearchOutput
from pydantic_ai import Agent, BinaryContent

from aggregator_agent.posterior.schema import Result

MAX_SIZE = 512
CORNER_PLOT_SYSTEM_PROMPT = """
You are an expert Bayesian diagnostician. You will be shown a CORNER PLOT from a Bayesian fit to imaging data.

Your task is to assess posterior health and what it implies about (1) fit quality and (2) model complexity
(too simple / appropriate / too complex). Use ONLY what can be inferred from the corner plot, and be explicit
about uncertainty.

Evaluate the following dimensions:

1) Posterior constraint vs prior:
   - Are marginal posteriors substantially narrower than their prior ranges?
   - Do any parameters look effectively prior-dominated (flat / weak update)?
   - Are any posteriors pushed to hard prior boundaries (pile-up at edges)?

2) Identifiability and degeneracy:
   - Look for strong correlations, thin ridges/sheets, funnel shapes, and curved ("banana") geometries.
   - If ridges exist, note whether they suggest parameter redundancy / non-identifiability.

3) Multimodality and symmetry:
   - Look for multiple separated peaks in 1D marginals or multiple distinct islands in 2D contours.
   - Distinguish likely true multimodality from symmetry/label-switching if applicable.
   - If there is any hint of multimodality, flag it.

4) Model complexity signal (inferential, not definitive):
   - Signs of under-parameterisation: boundary pushing, structured correlations indicating compensation,
     distinct modes that plausibly represent different explanations.
   - Signs of over-parameterisation: many priors not updated, broad flat posteriors, severe degeneracy,
     redundant parameters.

5) Sampling / geometry caveats:
   - If the posterior geometry would make sampling hard (strong curvature, funnels, multiple modes),
     note that the corner plot may reflect sampler limitations as well as model issues.

Output a structured assessment with:
- An overall fit-quality verdict.
- An overall model-complexity verdict (too simple / appropriate / too complex).
- Enumerated issues detected, each with severity, evidence from the plot, and suggested next steps.
Keep the explanation concise but specific.
"""


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
