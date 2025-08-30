#!/usr/bin/env python3
"""
MatchLines.py - Line and sentence matching utility

Implements the same matching methodology as generate_html_visualization.py
to find overlapping content between context and test markdown files.
"""

import re
from typing import Dict, List, Tuple, Set
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Global model instance for efficiency
_model = None

def get_embedding_model():
    """Get or initialize the sentence transformer model"""
    global _model
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast, lightweight model
    return _model

def normalize_line(line: str) -> str:
    """Normalize line by removing extra whitespace and converting to lowercase"""
    return ' '.join(line.strip().lower().split())

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences, treating newlines as sentence boundaries"""
    # First split by lines to respect line boundaries
    lines = text.split('\n')
    sentences = []

    for line in lines:
        line = line.strip()
        if not line:  # Skip empty lines
            continue

        # Split line into sentences, but keep line boundary constraint
        # Common sentence endings: . ! ? followed by space or end of line
        line_sentences = re.split(r'(?<=[.!?])\s+', line)

        for sentence in line_sentences:
            sentence = sentence.strip()
            if sentence and len(sentence.split()) >= 2:  # Skip very short sentences
                sentences.append(sentence)

    return sentences

def parse_markdown_line(line: str) -> Tuple[str, str, str]:
    """
    Parse a markdown line and return (element_type, content, html_tag)
    element_type: 'header', 'list_item', 'text', 'empty'
    content: the actual text content
    html_tag: the HTML tag to wrap it in
    """
    line = line.rstrip()

    if not line.strip():
        return 'empty', '', ''

    # Headers
    if line.startswith('#'):
        level = len(line) - len(line.lstrip('#'))
        level = min(6, max(1, level))  # Clamp between 1 and 6
        content = line.lstrip('# ').strip()
        return 'header', content, f'h{level}'

    # List items
    if line.lstrip().startswith(('* ', '- ', '+ ')):
        content = line.lstrip().lstrip('*-+ ').strip()
        return 'list_item', content, 'li'

    # Numbered lists
    if re.match(r'^\s*\d+\.\s+', line):
        content = re.sub(r'^\s*\d+\.\s+', '', line).strip()
        return 'list_item', content, 'li'

    # Regular text
    content = line.strip()
    return 'text', content, 'p'

def is_likely_header(line: str) -> bool:
    """Check if a line is likely a header based on formatting"""
    stripped = line.strip()
    if not stripped:
        return False

    # Check for markdown headers
    if stripped.startswith('#'):
        return True

    # Check for short lines that are likely titles
    words = stripped.split()
    if len(words) <= 5 and len(stripped) <= 50:
        # Check if it's all caps or title case
        if stripped.isupper() or stripped.istitle():
            return True
        # Check if it ends with colon (common for headers)
        if stripped.endswith(':'):
            return True
        # Check for common header patterns
        header_indicators = ['description', 'summary', 'overview', 'problem', 'solution', 'budget', 'timeline']
        if any(indicator in stripped.lower() for indicator in header_indicators):
            return True

    return False

def find_matched_lines_detailed(context_lines: List[str], test_lines: List[str]) -> Dict:
    """
    Find context line indices that were matched in test output using sentence-level semantic similarity.
    Returns detailed matching information including which sentences matched within each line.
    """
    matched_line_indices = set()
    line_sentence_matches = {}  # line_idx -> list of sentence indices that matched

    # Create sentence-to-line mapping for context
    context_sentences = []
    sentence_to_line = []
    line_to_sentences = {}  # line_idx -> list of sentence indices in that line

    sentence_idx = 0
    for line_idx, line in enumerate(context_lines):
        line_normalized = normalize_line(line)
        line_to_sentences[line_idx] = []

        if not line_normalized:  # Skip empty/whitespace lines
            continue

        # Split line into sentences
        line_sentences = split_into_sentences(line_normalized)
        for sentence in line_sentences:
            context_sentences.append(sentence)
            sentence_to_line.append(line_idx)
            line_to_sentences[line_idx].append(sentence_idx)
            sentence_idx += 1

    # Get all test sentences
    test_sentences = []
    for line in test_lines:
        line_normalized = normalize_line(line)
        if line_normalized:
            test_sentences.extend(split_into_sentences(line_normalized))

    if not context_sentences or not test_sentences:
        return {
            'matched_line_indices': set(),
            'line_sentence_matches': {},
            'line_to_sentences': line_to_sentences
        }

    # Get embeddings for sentences
    model = get_embedding_model()
    context_embeddings = model.encode(context_sentences, batch_size=32, show_progress_bar=False)
    test_embeddings = model.encode(test_sentences, batch_size=32, show_progress_bar=False)

    # Calculate cosine similarity matrix
    similarity_matrix = cosine_similarity(context_embeddings, test_embeddings)

    # Sequential greedy matching with order preservation and duplicate handling
    # Process test sentences in order, searching forward from last match position
    used_context_indices = set()
    last_matched_position = -1  # Track position of last matched input sentence

    for j in range(len(test_embeddings)):
        best_match_idx = None
        best_similarity = 0.0

        # Search strategy: first search from after last matched position to end,
        # then if nothing found, search from beginning to last matched position
        search_ranges = []

        # Range 1: From after last matched position to end
        if last_matched_position + 1 < len(context_sentences):
            search_ranges.append(range(last_matched_position + 1, len(context_sentences)))

        # Range 2: From beginning to last matched position (wrap-around for duplicates)
        if last_matched_position >= 0:
            search_ranges.append(range(0, last_matched_position + 1))
        elif last_matched_position == -1:
            # First sentence - search everything
            search_ranges.append(range(len(context_sentences)))

        # Search in the defined ranges
        for search_range in search_ranges:
            for i in search_range:
                if i in used_context_indices:
                    continue  # Skip already matched context sentences

                similarity = similarity_matrix[i][j]

                # Get threshold based on context line type
                line_idx = sentence_to_line[i]
                context_line = context_lines[line_idx]
                threshold = 0.60 if is_likely_header(context_line) else 0.70

                # Check if this is a good match and better than previous best
                if similarity >= threshold and similarity > best_similarity:
                    best_match_idx = i
                    best_similarity = similarity

            # If we found a good match in this range, stop searching
            if best_match_idx is not None:
                break

        # If we found a good match, mark it as used and update position
        if best_match_idx is not None:
            used_context_indices.add(best_match_idx)
            last_matched_position = best_match_idx  # Update position for next search
            line_idx = sentence_to_line[best_match_idx]
            matched_line_indices.add(line_idx)

            # Track which sentence within the line matched
            if line_idx not in line_sentence_matches:
                line_sentence_matches[line_idx] = []
            # Find the relative sentence index within the line
            line_sentence_idx = line_to_sentences[line_idx].index(best_match_idx)
            if line_sentence_idx not in line_sentence_matches[line_idx]:
                line_sentence_matches[line_idx].append(line_sentence_idx)

    return {
        'matched_line_indices': matched_line_indices,
        'line_sentence_matches': line_sentence_matches,
        'line_to_sentences': line_to_sentences
    }

def find_matched_lines(context_lines: List[str], test_lines: List[str]) -> Set[int]:
    """
    Find context line indices that were matched in test output using sentence-level semantic similarity.
    Returns set of context line indices (0-based).
    Matches at sentence level within line boundaries - no sentence crosses line boundaries.
    """
    detailed_result = find_matched_lines_detailed(context_lines, test_lines)
    return detailed_result['matched_line_indices']

def match_lines(context_markdown: str, test_markdown: str) -> Dict:
    """
    Match lines between context and test markdown using semantic similarity.

    Args:
        context_markdown: The context markdown text to search within
        test_markdown: The test markdown text to find matches for

    Returns:
        Dict containing:
        - context_lines: List of context lines
        - test_lines: List of test lines
        - context_sentences: List of sentences for each context line
        - test_sentences: List of sentences for each test line
        - overlap_line_numbers: List of context line indices (0-based) that match test content
        - line_match_details: Dict with match granularity for each matched line:
          {line_idx: {'type': 'full'|'partial', 'matched_sentences': [sentence_indices]}}
    """
    # Split into lines
    context_lines = [line.rstrip() for line in context_markdown.split('\n')]
    test_lines = [line.rstrip() for line in test_markdown.split('\n')]

    # Split each line into sentences
    context_sentences = []
    test_sentences = []

    for line in context_lines:
        element_type, content, _ = parse_markdown_line(line)
        if element_type != 'empty' and content:
            sentences = split_into_sentences(content)
            context_sentences.append(sentences)
        else:
            context_sentences.append([])

    for line in test_lines:
        element_type, content, _ = parse_markdown_line(line)
        if element_type != 'empty' and content:
            sentences = split_into_sentences(content)
            test_sentences.append(sentences)
        else:
            test_sentences.append([])

    # Get detailed matching information
    detailed_match = find_matched_lines_detailed(context_lines, test_lines)
    matched_line_indices = detailed_match['matched_line_indices']
    line_sentence_matches = detailed_match['line_sentence_matches']

    # Convert set to sorted list
    overlap_line_numbers = sorted(list(matched_line_indices))

    # Determine full vs partial matches for each line
    line_match_details = {}
    for line_idx in overlap_line_numbers:
        total_sentences_in_line = len(context_sentences[line_idx])
        matched_sentences = sorted(line_sentence_matches.get(line_idx, []))

        # Determine if it's a full or partial match
        if total_sentences_in_line == 0:
            match_type = 'full'  # Empty lines or single content treated as full
        elif len(matched_sentences) == total_sentences_in_line:
            match_type = 'full'
        else:
            match_type = 'partial'

        line_match_details[line_idx] = {
            'type': match_type,
            'matched_sentences': matched_sentences,
            'total_sentences': total_sentences_in_line
        }

    return {
        'context_lines': context_lines,
        'test_lines': test_lines,
        'context_sentences': context_sentences,
        'test_sentences': test_sentences,
        'overlap_line_numbers': overlap_line_numbers,
        'line_match_details': line_match_details
    }

# Example usage and testing
if __name__ == "__main__":
    # Test the function with sample markdown including multi-sentence lines
    context = """# Project Overview
This is a sample project description. It provides detailed information about the project scope and objectives.

## Key Features
* Feature A: Advanced analytics and reporting capabilities
* Feature B: Real-time processing. Handles data streaming efficiently.
* Feature C: User-friendly interface with modern design

The project aims to solve complex data problems. It uses modern technologies and best practices. The implementation follows industry standards."""

    test = """# Overview
This project focuses on data analytics.

Key capabilities include:
- Real-time processing of data
- Analytics dashboard
- Modern interface design

The solution uses modern technologies effectively."""

    result = match_lines(context, test)

    print("Context lines with detailed match information:")
    for i, line in enumerate(result['context_lines']):
        if i in result['overlap_line_numbers']:
            details = result['line_match_details'][i]
            match_info = f" ✓ {details['type'].upper()}"
            if details['type'] == 'partial':
                match_info += f" (sentences {details['matched_sentences']} of {details['total_sentences']})"
            else:
                match_info += f" (all {details['total_sentences']} sentences)" if details['total_sentences'] > 1 else ""
        else:
            match_info = ""

        print(f"{i:2d}: {line}{match_info}")

        # Show sentences for matched lines
        if i in result['overlap_line_numbers'] and result['context_sentences'][i]:
            for j, sentence in enumerate(result['context_sentences'][i]):
                sentence_marker = "    →" if j in result['line_match_details'][i]['matched_sentences'] else "     "
                print(f"{sentence_marker} [{j}] {sentence}")

    print(f"\nTest lines:")
    for i, line in enumerate(result['test_lines']):
        print(f"{i:2d}: {line}")

    print(f"\nMatch Summary:")
    print(f"Overlapping context line numbers: {result['overlap_line_numbers']}")
    print(f"Total context lines: {len([l for l in result['context_lines'] if l.strip()])}")
    print(f"Matched context lines: {len(result['overlap_line_numbers'])}")
    print(f"Match percentage: {len(result['overlap_line_numbers'])/len([l for l in result['context_lines'] if l.strip()]) * 100:.1f}%")

    print(f"\nDetailed Match Analysis:")
    for line_idx in result['overlap_line_numbers']:
        details = result['line_match_details'][line_idx]
        print(f"Line {line_idx}: {details['type']} match ({len(details['matched_sentences'])}/{details['total_sentences']} sentences)")
        if details['type'] == 'partial':
            print(f"  Matched sentence indices: {details['matched_sentences']}")
            for sent_idx in details['matched_sentences']:
                print(f"    [{sent_idx}] {result['context_sentences'][line_idx][sent_idx]}")
