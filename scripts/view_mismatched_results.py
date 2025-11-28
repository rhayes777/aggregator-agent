"""Interactive viewer for non-good classification results.

Reads a CSV (default: results.csv) and for every row where either
`expected_category` or `predicted_category` is not "Good", prints the
categories/descriptions and displays the associated image
(`data/initial_lens_model/{id}.png`). The image window waits for a button
press before closing and advancing to the next result.
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt


def is_good(category: str) -> bool:
    return category.strip().lower() == "good"


def read_rows(csv_path: Path) -> Iterable[Dict[str, str]]:
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def show_image(image_path: Path, title: str) -> None:
    """Display an image and wait until user presses a key or clicks to continue."""
    img = plt.imread(image_path)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)
    ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()

    def _close_event(_event) -> None:
        plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", _close_event)
    fig.canvas.mpl_connect("button_press_event", _close_event)

    print("Press any key or click in the image window to continue...")
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="View results where either category is not 'Good'."
    )
    parser.add_argument(
        "--csv",
        default="results.csv",
        type=Path,
        help="Path to results CSV (default: results.csv)",
    )
    parser.add_argument(
        "--image-root",
        default=Path("data/initial_lens_model"),
        type=Path,
        help="Root directory containing <id>.png images.",
    )
    args = parser.parse_args()

    csv_path: Path = args.csv
    image_root: Path = args.image_root

    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    for row in read_rows(csv_path):
        expected = row.get("expected_category", "")
        predicted = row.get("predicted_category", "")
        expected_desc = row.get("expected_description", "").strip()
        predicted_desc = row.get("predicted_description", "").strip()
        sample_id = row.get("id", "").strip()

        if is_good(expected) and is_good(predicted):
            continue

        print("-" * 60)
        print(f"ID: {sample_id}")
        print(f"Expected: {expected}")
        print(f"Predicted: {predicted}")
        print(f"Expected description: {expected_desc}")
        print(f"Predicted description: {predicted_desc}")

        image_path = image_root / f"{sample_id}.png"
        if not image_path.exists():
            print(f"[missing image] {image_path}")
            continue

        try:
            show_image(image_path, title=sample_id)
        except Exception as exc:  # pragma: no cover - convenience for runtime issues
            print(f"Failed to display {image_path}: {exc}")


if __name__ == "__main__":
    main()
