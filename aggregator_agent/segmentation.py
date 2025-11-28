from PIL import Image
from pathlib import Path

directory = Path(__file__).parents[1]
segmentation_directory = directory / "data/segmentation"

for path in segmentation_directory.iterdir():
    # Load image (RGB assumed)
    img0 = Image.open(path / "rgb_0.png")
    w, h = img0.size

    # Crop central 100x100
    crop_size = 100
    half = crop_size // 2

    center_x = w // 2
    center_y = h // 2

    left = center_x - half
    right = center_x + half
    top = center_y - half
    bottom = center_y + half

    # Perform the crop
    crop = img0.crop((left, top, right, bottom))

    # Save output
    crop.save(path / "rgb_zoom.png")

    print("Saved:", path / "rgb_zoom.png")
