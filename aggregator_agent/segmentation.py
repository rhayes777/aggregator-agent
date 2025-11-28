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
    try:
        image_path = path / "rgb_zoom.png"

        with image_path.open("rb") as f:
            image_bytes = f.read()

        original_image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")

        mask = agent.run_sync(
            [
                BinaryContent(
                    data=image_bytes,
                    media_type="image/png"  # or image/jpeg etc. depending on the file
                ),
            ]
        ).response.images[0]
        image = Image.open(io.BytesIO(mask.data)).convert("RGBA")

        # Resize (no cropping) to match the source image so the overlay lines up.
        image = image.resize(original_image.size, Image.NEAREST)

        # Make black pixels fully transparent and keep coloured regions translucent.
        translucent_alpha = 140
        pixels = []
        for r, g, b, _ in image.getdata():
            if r == 0 and g == 0 and b == 0:
                pixels.append((r, g, b, 0))
            else:
                pixels.append((r, g, b, translucent_alpha))
        image.putdata(pixels)

        mask_path = path / "mask.png"
        image.save(str(mask_path))
        print("Saved mask to:", mask_path)

        overlay = Image.alpha_composite(original_image, image)
        overlay_path = path / "mask_overlay.png"
        overlay.save(str(overlay_path))
        print("Saved overlay to:", overlay_path)
    except Exception as e:
        print("Error processing", path, ":", e)
