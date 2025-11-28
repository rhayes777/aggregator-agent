from pathlib import Path
from csv import DictReader

directory = Path(__file__).parent
data_directory = directory / "data"


with open(directory / "image_analysis.csv") as f:
    for row in DictReader(f):
        print(row)
