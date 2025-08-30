# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a context compression optimization experiment for an Agentic RAG system. The project aims to improve GPT-4o-mini's performance on context compression tasks by using DSPy GEPA (Genetic-Pareto) optimization to enhance prompts that were originally designed for GPT-4o.

The core problem: GPT-4o-mini frequently fails at contextual compression (outputting "NO_OUTPUT") where GPT-4o succeeds, leading to production fallbacks and suboptimal user experience.

## Dataset Structure

The repository contains a curated dataset of failed contextual compressions:
- `data/observations/`: 1,700+ JSON files containing failed GPT-4o-mini contextual compressions from LangFuse traces
- `data/gpt-4o/`: Corresponding successful GPT-4o compressions for ~300 of the failed cases, showing target behavior

Each observation file contains:
- `id`: Unique identifier from LangFuse trace
- `input`: Array with system prompt and user query containing document context and search query
- `output`: Model response (typically "NO_OUTPUT" for failed cases)
- Metadata including model parameters, timestamps, and usage statistics

Each GPT-4o success file contains:
- `observation_id`: Matching ID from observations
- `original_output`: Original failed output ("NO_OUTPUT")
- `compression_successful`: Whether GPT-4o succeeded (true)
- `output`: The successful contextual compression text

## Context Compression Task

The contextual compression prompt extracts relevant portions of documents based on search queries. Key requirements:
1. Extract text AS IS without modification
2. Include headers and related sections
3. Preserve exact formatting and content
4. Output "NO_OUTPUT" if no relevant information exists

The current prompt performs well with GPT-4o but poorly with GPT-4o-mini due to model capability differences.

## Optimization Approaches

The project implements multiple optimization strategies for context compression prompts:

### 1. DSPy GEPA Optimization (Primary)
Uses DSPy GEPA (Genetic-Pareto) optimization to:
1. Generate prompt variants through genetic algorithms
2. Evaluate performance on the curated dataset
3. Find Pareto-optimal solutions balancing accuracy and cost
4. Improve GPT-4o-mini performance on contextual compression

### 2. TextGrad Optimization (Alternative)
Uses TextGrad's textual gradient descent to:
1. Optimize prompts through natural language feedback
2. Apply gradient-based refinement to system instructions
3. Provide interpretable optimization with textual gradients
4. Enable fine-grained prompt adjustments

### 3. Hybrid TextGrad+GEPA Optimization (Advanced)
Combines both approaches in a two-stage pipeline:
1. GEPA provides scientifically optimized starting prompt
2. TextGrad applies gradient-based refinement to GEPA results
3. Enables incremental improvement on existing optimizations
4. Tests whether TextGrad can enhance GEPA-optimized prompts

## Setup and Development

### Initial Setup
```bash
make setup
```
This will:
- Create a virtual environment using `uv`
- Install all project dependencies
- Set up the development environment

### Environment Configuration
1. Copy `.env.template` to `.env`
2. Add your OpenAI API key to `.env`

### Optimization Commands
```bash
# DSPy GEPA Optimization
make optimize                    # Run DSPy GEPA optimization (primary method)

# TextGrad Optimization  
make optimize-textgrad          # Run TextGrad optimization from base prompt
make optimize-textgrad-gepa     # Run TextGrad starting from latest GEPA prompt

# Testing Optimized Prompts
make test-gepa                  # Test latest GEPA-optimized prompt
make test-textgrad             # Test latest TextGrad-optimized prompt  
make test-textgrad-gepa        # Test latest TextGrad+GEPA hybrid prompt

# Process Management
make kill-optimizer            # Kill DSPy GEPA processes
make kill-textgrad            # Kill TextGrad optimizer processes
make kill-textgrad-gepa       # Kill TextGrad+GEPA optimizer processes
make kill-test-gepa           # Kill GEPA test processes
make kill-test-textgrad       # Kill TextGrad test processes
make kill-test-textgrad-gepa  # Kill TextGrad+GEPA test processes
```

### Common Development Commands
```bash
make help           # Show all available commands
make install        # Install dependencies only
make install-dev    # Install with development dependencies
make test           # Run tests
make format         # Format code with black and isort
make lint           # Run linting
make check          # Type checking with mypy
make notebook       # Start Jupyter Lab
make data-check     # Verify data directory structure
make clean          # Clean build artifacts
make clear-cache    # Clear DSPy cache contents
```

### Project Structure
- `src/context_compression_experiments/` - Main package code
- `data/observations/` - Failed GPT-4o-mini compressions (1,700+ files)
- `data/gpt-4o/` - Successful GPT-4o compressions (~300 files)
- `data/results/` - Optimization results and experiment artifacts
- `data/tests/` - Testing results for optimized prompts
- `scripts/` - Optimization and testing scripts
- `tests/` - Unit test files
- `.venv/` - Virtual environment (created by setup)

### Scripts Directory
- `dspy_gepa_optimizer.py` - DSPy GEPA optimization script
- `textgrad_optimizer.py` - TextGrad optimization from base prompt
- `textgrad_latest_gepa_optimizer.py` - TextGrad optimization starting from GEPA
- `test_latest_gepa_prompt.py` - Test GEPA-optimized prompts
- `test_latest_textgrad_prompt.py` - Test TextGrad-optimized prompts
- `test_latest_textgrad_gepa_prompt.py` - Test TextGrad+GEPA hybrid prompts

## Development Environment

- Python 3.9+ with `uv` for dependency management
- VS Code with spell checker configured for domain terms (Agentic, GEPA, langgraph, etc.)
- Dataset stored in JSON format for easy processing with ML frameworks
- Jupyter Lab for interactive development and experimentation
- Git repository for version control of experiments and results

## Optimization Workflow

### Complete Experimental Pipeline

The project supports a comprehensive optimization and testing pipeline:

```bash
# 1. Run all optimization methods
make optimize                    # Generate GEPA baseline
make optimize-textgrad          # Generate TextGrad from scratch  
make optimize-textgrad-gepa     # Generate TextGrad refinement of GEPA

# 2. Test all methods for performance comparison
make test-gepa                  # Test GEPA performance
make test-textgrad             # Test TextGrad performance
make test-textgrad-gepa        # Test hybrid performance

# 3. Compare results across optimization methods
# Results saved in data/tests/ with clear naming for analysis
```

### Results Directory Structure

Optimization results are saved in timestamped directories under `data/results/`:
- `gepa_context_compression_{timestamp}/` - GEPA optimization results
- `textgrad_context_compression_{timestamp}/` - TextGrad optimization results
- `textgrad_gepa_context_compression_{timestamp}/` - Hybrid optimization results

Testing results are saved under `data/tests/`:
- `gpt-4o-mini-test-{timestamp}/` - GEPA prompt testing results
- `textgrad-gpt-4o-mini-test-{timestamp}/` - TextGrad prompt testing results
- `textgrad-gepa-gpt-4o-mini-test-{timestamp}/` - Hybrid prompt testing results

### Key Dependencies

The project uses several optimization libraries:
- `dspy-ai>=2.4.0` - DSPy framework for GEPA optimization
- `textgrad>=0.1.4` - TextGrad framework for textual gradient descent
- `openai>=1.0.0` - OpenAI API for model interactions
- `weave>=0.50.0` - W&B Weave for experiment tracking
- `wandb>=0.16.0` - Weights & Biases for logging

### Environment Variables Required

Configure these in your `.env` file:
- `OPENAI_API_KEY` - Required for all optimization and testing
- `WANDB_API_KEY` - Optional, for experiment tracking  
- `WANDB_PROJECT` - Optional, defaults to "context-compression-experiments"
- `DSPY_CACHEDIR` - Optional, for custom cache directory

## Common Tasks

When working on this project, typical tasks include:
- Running optimization experiments with different methods (GEPA, TextGrad, hybrid)
- Testing optimized prompts against observation datasets
- Analyzing performance improvements across optimization approaches
- Comparing success rates between GPT-4o vs GPT-4o-mini on context compression
- Processing and analyzing optimization results and metrics
- Managing long-running optimization processes with kill commands
- Reviewing experiment logs and tracking optimization progress