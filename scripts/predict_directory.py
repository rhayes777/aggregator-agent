#!/usr/bin/env Python
"""
Use a VLM to categorise images produced by lens modelling in a given directory.
"""
import csv
from argparse import ArgumentParser
from pathlib import Path

from aggregator_agent.image_agent import categorise

parser = ArgumentParser("Read images from a directory and assess the quality of the lensing")

parser.add_argument(
    "directory",
    type=Path,
)
parser.add_argument(
    "--output",
    type=Path,
    default=None,
)

args = parser.parse_args()

output_filename = args.output or args.directory.with_name(f"{args.directory.stem}_categorised.csv")

with output_filename.open("w") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "category", "description"])
    for path in args.directory.iterdir():
        result = categorise(path)
        writer.writerow([path.stem, result.category, result.description])
