from argparse import ArgumentParser
from io import BytesIO

from pathlib import Path

from PIL import Image
from pydantic_ai import Agent, BinaryContent

MAX_SIZE = 512

parser = ArgumentParser()
parser.add_argument(
    "image",
    type=Path,
    help="Path to the image file to be analyzed.",
)
parser.add_argument(
    "--max-size",
    type=int,
    default=MAX_SIZE,
    help=f"Maximum size (in pixels) for the longest side of the image. Default is {MAX_SIZE}.",
)
args = parser.parse_args()

SYSTEM_PROMPT = """
I will show you the corner plot resultant from a bayesian fit to imaging data.

Describe the quality of the fit.
"""



agent = Agent(
    model="gpt-5",
    instructions=SYSTEM_PROMPT,
)

with Image.open(args.image) as img:
    img = img.convert("RGB")  # ensure consistent format

    # Resize in-place, preserving aspect ratio
    img.thumbnail(
        (args.max_size, args.max_size),
        Image.Resampling.LANCZOS,
    )

    buffer = BytesIO()
    img.save(buffer, format="PNG")
    # also show image for sanity
    img.show()

    image_bytes = buffer.getvalue()

result = agent.run_sync(
    [
        BinaryContent(
            data=image_bytes,
            media_type="image/png",  # or image/jpeg etc. depending on the file
        ),
    ]
).output

print(result)
