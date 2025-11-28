from pathlib import Path

from aggregator_agent.segmentation import process_image

directory = Path(__file__).parents[1]
segmentation_directory = directory / "data/segmentation"

for path in list(segmentation_directory.iterdir()):
    print("Processing:", path)
    try:
        process_image(path / "rgb_zoom.png")
    except Exception as e:
        print("Error processing", path, ":", e)
