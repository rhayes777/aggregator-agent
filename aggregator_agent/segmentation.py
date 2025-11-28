import base64
import io
from pathlib import Path

from PIL import Image
from openai import OpenAI

directory = Path(__file__).parents[1]
segmentation_directory = directory / "data/segmentation"

client = OpenAI()

INSTRUCTIONS = """
You are an expert astronomer analysing an image of a gravitational lens.

Your task is to identify and segment light from different components in the image, such as the lens galaxy, source galaxy, and any nearby objects.
Mask any pixels containing the lens galaxy with red, the source galaxy with green, and other objects with blue.
"""

TARGET_SIZE = (1024, 1024)
TARGET_SIZE_STR = f"{TARGET_SIZE[0]}x{TARGET_SIZE[1]}"

for path in segmentation_directory.iterdir():
    print("Processing:", path)
    try:
        image_path = path / "rgb_zoom.png"

        with image_path.open("rb") as f:
            image_bytes = f.read()

        original_image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
        original_image = original_image.resize(TARGET_SIZE, Image.LANCZOS)

        # Generate the mask via OpenAI Responses API using the image generation tool.
        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        response = client.responses.create(
            model="gpt-5",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": INSTRUCTIONS.strip()},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{b64_image}",
                        },
                    ],
                }
            ],
            tools=[
                {
                    "type": "image_generation",
                    "size": TARGET_SIZE_STR,
                    "quality": "low",
                }
            ],
        )

        image_data = [
            output.result
            for output in response.output
            if output.type == "image_generation_call"
        ]

        if not image_data:
            raise RuntimeError("No image returned from OpenAI image generation tool.")

        mask_bytes = base64.b64decode(image_data[0])
        image = Image.open(io.BytesIO(mask_bytes)).convert("RGBA")

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
