from io import BytesIO

from pathlib import Path

from PIL import Image
from pydantic_ai import Agent, BinaryContent

MAX_SIZE = 512
CORNER_PLOT_SYSTEM_PROMPT = """
I will show you the corner plot resultant from a bayesian fit to imaging data.

Describe the quality of the fit.
"""


class PosteriorFitAnalysis:
    def __init__(self, fit_path: Path, max_image_size: int = MAX_SIZE):
        self.fit_path = fit_path
        self.max_image_size = max_image_size

    @property
    def corner_plot_path(self) -> Path:
        return self.fit_path / "image/search/corner_anesthetic.png"

    @property
    def image_bytes(self) -> bytes:
        with Image.open(self.corner_plot_path) as img:
            img = img.convert("RGB")  # ensure consistent format

            # Resize in-place, preserving aspect ratio
            img.thumbnail(
                (self.max_image_size, self.max_image_size),
                Image.Resampling.LANCZOS,
            )

            buffer = BytesIO()
            img.save(buffer, format="PNG")
            return buffer.getvalue()

    def corner_plot_analysis(self) -> str:
        agent = Agent(
            model="gpt-5",
            instructions=CORNER_PLOT_SYSTEM_PROMPT,
        )

        result = agent.run_sync(
            [
                BinaryContent(
                    data=self.image_bytes,
                    media_type="image/png",  # or image/jpeg etc. depending on the file
                ),
            ]
        ).output

        return result
