#!/usr/bin/env python3
"""
generate_coverage_image.py - Generate an image representing test coverage over context.
"""

import sys
from pathlib import Path
from PIL import Image

# Add the script's directory to the Python path to allow importing MatchLines
sys.path.append(str(Path(__file__).parent))

from MatchLines import match_lines

# Sample markdown strings
context_markdown = """# Project Overview
This is a sample project description. It provides detailed information about the project scope and objectives.

## Key Features
* Feature A: Advanced analytics and reporting capabilities
* Feature B: Real-time processing. Handles data streaming efficiently.
* Feature C: User-friendly interface with modern design

The project aims to solve complex data problems. It uses modern technologies and best practices. The implementation follows industry standards."""

test_markdown = """# Overview
This project focuses on data analytics.

Key capabilities include:
- Real-time processing of data
- Analytics dashboard
- Modern interface design

The solution uses modern technologies effectively."""

def generate_coverage_image(context, test, output_path="coverage.png"):
    """
    Generates an image representing the coverage of test_markdown over context_markdown.
    """
    match_result = match_lines(context, test)
    context_lines = match_result['context_lines']
    overlap_lines = match_result['overlap_line_numbers']

    line_height = 10
    line_width = 2
    image_height = line_height
    image_width = len(context_lines) * line_width

    img = Image.new('RGB', (image_width, image_height), color='black')
    pixels = img.load()

    for i, line in enumerate(context_lines):
        color = (255, 0, 0) if i in overlap_lines else (0, 0, 0)
        for x in range(line_width):
            for y in range(line_height):
                pixels[i * line_width + x, y] = color

    img.save(output_path)
    print(f"Coverage image saved to {output_path}")

if __name__ == "__main__":
    generate_coverage_image(context_markdown, test_markdown)
