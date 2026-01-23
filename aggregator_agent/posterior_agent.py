from argparse import ArgumentParser

from pathlib import Path

from pydantic_ai import Agent, BinaryContent

parser = ArgumentParser()
parser.add_argument(
    "image",
    type=Path,
    help="Path to the image file to be analyzed.",
)
args = parser.parse_args()

SYSTEM_PROMPT = """
I will show you the corner plot resultant from a bayesian fit to imaging data.

Describe the quality of the fit.
"""

agent = Agent(
    model='gpt-5',
    instructions=SYSTEM_PROMPT,
)

with args.image.open("rb") as f:
    image_bytes = f.read()

result = agent.run_sync(
    [
        BinaryContent(
            data=image_bytes,
            media_type="image/png"  # or image/jpeg etc. depending on the file
        ),
    ]
).output

print(result)
