#!/usr/bin/env python

from pathlib import Path
from argparse import ArgumentParser

from aggregator_agent.image_agent import categorise

directory = Path(__file__).parents[1]
data_directory = directory / "data"
initial_lens_model_directory = data_directory / "initial_lens_model"

parser = ArgumentParser()

parser.add_argument("id")

args = parser.parse_args()

print(categorise((initial_lens_model_directory / args.id).with_suffix(".png")))
