import argparse
from pathlib import Path

from aggregator_agent.segmentation import process_image

parser = argparse.ArgumentParser(description="Segment One Script")
parser.add_argument("image_path", type=Path, help="Input file path")

args = parser.parse_args()

process_image(args.image_path)
