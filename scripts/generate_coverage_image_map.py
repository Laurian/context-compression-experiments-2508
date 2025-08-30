#!/usr/bin/env python3
"""
generate_coverage_image_map.py - Generates a coverage map image from test results.

This script iterates through JSON files in a given test data folder, calculates
the test coverage for each, and generates a single PNG image to visualize the results.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from PIL import Image

# Add the script's directory to the Python path to allow importing MatchLines
sys.path.append(str(Path(__file__).parent))

from MatchLines import match_lines

# Web colors dictionary (8-bit RGB values)
WEB_COLORS = {
    'red': (255, 0, 0),
    'green': (0, 255, 0),
    'blue': (0, 0, 255),
    'yellow': (255, 255, 0),
    'cyan': (0, 255, 255),
    'magenta': (255, 0, 255),
    'orange': (255, 165, 0),
    'purple': (128, 0, 128),
    'pink': (255, 192, 203),
    'lime': (0, 255, 0),
    'indigo': (75, 0, 130),
    'violet': (238, 130, 238),
    'brown': (165, 42, 42),
    'gray': (128, 128, 128),
    'grey': (128, 128, 128),
    'black': (0, 0, 0),
    'white': (255, 255, 255),
    'navy': (0, 0, 128),
    'maroon': (128, 0, 0),
    'olive': (128, 128, 0),
    'teal': (0, 128, 128),
    'silver': (192, 192, 192),
    'gold': (255, 215, 0),
    'coral': (255, 127, 80),
    'salmon': (250, 128, 114),
    'khaki': (240, 230, 140),
    'plum': (221, 160, 221),
    'crimson': (220, 20, 60),
    'turquoise': (64, 224, 208)
}

def extract_context_from_input(input_content: str) -> str:
    """Extracts context from <context> tags in the input string."""
    match = re.search(r'<context>(.*?)</context>', input_content, re.DOTALL)
    return match.group(1).strip() if match else ""

def generate_coverage_map(test_folder: Path, output_path: str = "coverage_map.png", color: str = "red"):
    """
    Generates a coverage map for all JSON results in a test folder.

    Args:
        test_folder: Path to the test folder containing results
        output_path: Path where to save the output image
        color: Color name for coverage highlighting (default: 'red')
    """
    results_folder = test_folder / "results"
    if not results_folder.is_dir():
        print(f"Error: 'results' subfolder not found in {test_folder}")
        return

    observations_folder = Path(__file__).parent.parent / "data" / "observations"
    if not observations_folder.is_dir():
        print(f"Error: 'observations' folder not found at {observations_folder}")
        return

    json_files = sorted(list(results_folder.glob("*.json")))
    if not json_files:
        print(f"No JSON files found in {results_folder}")
        return

    all_results = []
    max_lines = 0

    print(f"Found {len(json_files)} JSON files. Processing...")

    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)

        observation_id = data.get("id")
        if not observation_id:
            continue

        observation_file = observations_folder / f"{observation_id}.json"
        if not observation_file.exists():
            continue

        with open(observation_file, 'r') as f:
            observation_data = json.load(f)

        input_content = observation_data.get("input", [{}])[0].get("content", "")
        context = extract_context_from_input(input_content)
        output = data.get("output", "")

        if not context or not output:
            continue

        match_result = match_lines(context, output)
        all_results.append(match_result)
        max_lines = max(max_lines, len(match_result['context_lines']))

    if not all_results:
        print("No valid data found in JSON files.")
        return

    # Get the RGB color value
    if color.lower() in WEB_COLORS:
        coverage_color = WEB_COLORS[color.lower()]
    else:
        print(f"Warning: Color '{color}' not found in web colors. Using red as default.")
        coverage_color = WEB_COLORS['red']

    line_height = 10
    line_width = 2
    image_height = len(all_results) * line_height
    image_width = max_lines * line_width

    img = Image.new('RGB', (image_width, image_height), color='black')
    pixels = img.load()

    for row, result in enumerate(all_results):
        context_lines = result['context_lines']
        overlap_lines = result['overlap_line_numbers']

        for i in range(len(context_lines)):
            pixel_color = coverage_color if i in overlap_lines else (50, 50, 50) # Dark grey for non-overlap
            for x in range(line_width):
                for y in range(line_height):
                    pixels[i * line_width + x, row * line_height + y] = pixel_color

    img.save(output_path)
    print(f"Coverage map image saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a coverage map from test results.")
    parser.add_argument("test_folder", type=str, help="Path to the test folder under data/tests/")
    parser.add_argument("--color", "-c", type=str, default="red",
                        help=f"Color for coverage highlighting. Available colors: {', '.join(sorted(WEB_COLORS.keys()))} (default: red)")
    args = parser.parse_args()

    # Construct the full path to the test folder
    base_path = Path(__file__).parent.parent / "data" / "tests"
    test_folder_path = base_path / args.test_folder

    if not test_folder_path.is_dir():
        print(f"Error: Test folder not found at {test_folder_path}")
        sys.exit(1)

    generate_coverage_map(test_folder_path, color=args.color)
