import io
from pathlib import Path

from PIL import Image
from pydantic_ai import Agent, ImageGenerationTool, BinaryContent
from pydantic_ai.models.openai import OpenAIResponsesModel

directory = Path(__file__).parents[1]
segmentation_directory = directory / "data/segmentation"

agent = Agent(
    model=OpenAIResponsesModel('gpt-5'),
    builtin_tools=[
        ImageGenerationTool(
            size="1024x1024",
            quality="low",
        )
    ],
    instructions="""
You are an expert astronomer analysing an image of a gravitational lens.

Your task is to identify and segment light from different components in the image, such as the lens galaxy, source galaxy, and any nearby objects.
Mask any pixels containing the lens galaxy with red, the source galaxy with green, and other objects with blue.
"""
)

for path in segmentation_directory.iterdir():
    print("Processing:", path)
    image_path = path / "rgb_zoom.png"

    with image_path.open("rb") as f:
        image_bytes = f.read()

    mask = agent.run_sync(
        [
            BinaryContent(
                data=image_bytes,
                media_type="image/png"  # or image/jpeg etc. depending on the file
            ),
        ]
    ).response.images[0]
    image = Image.open(io.BytesIO(mask.data))
    image.save(str(path / "mask.png"))
    print("Saved mask to:", path / "mask.png")