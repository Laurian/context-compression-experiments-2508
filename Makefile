.PHONY: help setup install install-dev clean test lint format check notebook jupyter data-check activate optimize optimize-textgrad optimize-textgrad-gepa test-gepa test-textgrad test-textgrad-gepa kill-optimizer kill-test-gepa kill-textgrad kill-test-textgrad kill-textgrad-gepa kill-test-textgrad-gepa clear-cache

# Default target
help:
	@echo "Available commands:"
	@echo "  setup       - Initial project setup (create venv and install dependencies)"
	@echo "  install     - Install project dependencies"
	@echo "  install-dev - Install project with development dependencies"
	@echo "  clean       - Remove build artifacts and cache files"
	@echo "  test        - Run tests with pytest"
	@echo "  lint        - Run linting with flake8"
	@echo "  format      - Format code with black and isort"
	@echo "  check       - Run type checking with mypy"
	@echo "  notebook    - Start Jupyter Lab"
	@echo "  jupyter     - Alias for notebook"
	@echo "  data-check  - Verify data directory structure and sample files"
	@echo "  optimize    - Run DSPy GEPA optimization on context compression"
	@echo "  optimize-textgrad - Run TextGrad optimization on context compression"
	@echo "  optimize-textgrad-gepa - Run TextGrad optimization starting from latest GEPA prompt"
	@echo "  test-gepa   - Test latest GEPA-optimized prompt against all observations"
	@echo "  test-textgrad - Test latest TextGrad-optimized prompt against all observations"
	@echo "  test-textgrad-gepa - Test latest TextGrad+GEPA-optimized prompt against all observations"
	@echo "  kill-optimizer - Kill any running optimizer processes"
	@echo "  kill-test-gepa - Kill any running GEPA test processes"
	@echo "  kill-textgrad - Kill any running TextGrad optimizer processes"
	@echo "  kill-test-textgrad - Kill any running TextGrad test processes"
	@echo "  kill-textgrad-gepa - Kill any running TextGrad+GEPA optimizer processes"
	@echo "  kill-test-textgrad-gepa - Kill any running TextGrad+GEPA test processes"
	@echo "  clear-cache - Clear DSPy cache contents (keep .gitkeep files)"
	@echo "  activate    - Show command to activate virtual environment"

# Initial project setup
setup:
	@echo "Setting up project..."
	@if [ ! -d ".venv" ]; then \
		echo "Creating virtual environment..."; \
		uv venv; \
	fi
	@echo "Installing dependencies..."
	@uv sync
	@echo "Setup complete! Run 'make activate' to see how to activate the environment."

# Install dependencies
install:
	uv sync --no-dev

# Install with development dependencies
install-dev:
	uv sync

# Clean up build artifacts and cache
clean:
	@echo "Cleaning up build artifacts and cache..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf __pycache__/
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +

# Run tests
test:
	@echo "Running tests..."
	uv run pytest

# Run linting
lint:
	@echo "Running linting..."
	uv run flake8 src/ tests/

# Format code
format:
	@echo "Formatting code..."
	uv run black src/ tests/
	uv run isort src/ tests/

# Type checking
check:
	@echo "Running type checking..."
	uv run mypy src/

# Start Jupyter Lab
notebook:
	@echo "Starting Jupyter Lab..."
	uv run jupyter lab

# Alias for notebook
jupyter: notebook

# Verify data directory structure
data-check:
	@echo "Checking data directory structure..."
	@if [ -d "data" ]; then \
		echo " data/ directory exists"; \
		echo "  - observations: $$(find data/observations -name "*.json" | wc -l | tr -d ' ') files"; \
		echo "  - gpt-4o: $$(find data/gpt-4o -name "*.json" | wc -l | tr -d ' ') files"; \
	else \
		echo " data/ directory not found"; \
	fi
	@if [ -f ".env" ]; then \
		echo " .env file exists"; \
	else \
		echo " .env file not found - copy .env.template to .env and configure"; \
	fi

# Run DSPy GEPA optimization
optimize:
	@echo "Running DSPy GEPA optimization..."
	@echo "This will optimize the context compression prompt using genetic algorithms."
	@echo "Make sure you have configured your .env file with API keys."
	uv run python scripts/dspy_gepa_optimizer.py

# Run TextGrad optimization
optimize-textgrad:
	@echo "Running TextGrad optimization..."
	@echo "This will optimize the context compression prompt using textual gradient descent."
	@echo "Make sure you have configured your .env file with API keys."
	uv run python scripts/textgrad_optimizer.py

# Run TextGrad optimization starting from latest GEPA prompt
optimize-textgrad-gepa:
	@echo "Running TextGrad optimization starting from latest GEPA prompt..."
	@echo "This will use the latest GEPA-optimized prompt as starting point for TextGrad."
	@echo "Make sure you have run 'make optimize' first to generate a GEPA baseline."
	@echo "Make sure you have configured your .env file with API keys."
	uv run python scripts/textgrad_latest_gepa_optimizer.py

# Test latest GEPA-optimized prompt
test-gepa:
	@echo "Testing latest GEPA-optimized prompt..."
	@echo "This will test the optimized prompt against all observations using GPT-4o-mini."
	@echo "Make sure you have configured your OPENAI_API_KEY in .env file."
	uv run python scripts/test_latest_gepa_prompt.py

# Test latest TextGrad-optimized prompt
test-textgrad:
	@echo "Testing latest TextGrad-optimized prompt..."
	@echo "This will test the optimized prompt against all observations using GPT-4o-mini."
	@echo "Make sure you have configured your OPENAI_API_KEY in .env file."
	uv run python scripts/test_latest_textgrad_prompt.py

# Test latest TextGrad+GEPA-optimized prompt
test-textgrad-gepa:
	@echo "Testing latest TextGrad+GEPA-optimized prompt..."
	@echo "This will test the TextGrad-refined GEPA prompt against all observations using GPT-4o-mini."
	@echo "Make sure you have run 'make optimize-textgrad-gepa' first to generate a hybrid result."
	@echo "Make sure you have configured your OPENAI_API_KEY in .env file."
	uv run python scripts/test_latest_textgrad_gepa_prompt.py

# Kill any running optimizer processes
kill-optimizer:
	@echo "Killing any running optimizer processes..."
	@pkill -f "dspy_gepa_optimizer.py" || echo "No optimizer processes found"
	@pkill -f "make optimize" || echo "No make optimize processes found"
	@echo "Optimizer processes killed"

# Kill any running GEPA test processes
kill-test-gepa:
	@echo "Killing any running GEPA test processes..."
	@pkill -f "test_latest_gepa_prompt.py" || echo "No GEPA test processes found"
	@pkill -f "make test-gepa" || echo "No make test-gepa processes found"
	@echo "GEPA test processes killed"

# Kill any running TextGrad optimizer processes
kill-textgrad:
	@echo "Killing any running TextGrad optimizer processes..."
	@pkill -f "textgrad_optimizer.py" || echo "No TextGrad optimizer processes found"
	@pkill -f "make optimize-textgrad" || echo "No make optimize-textgrad processes found"
	@echo "TextGrad optimizer processes killed"

# Kill any running TextGrad test processes
kill-test-textgrad:
	@echo "Killing any running TextGrad test processes..."
	@pkill -f "test_latest_textgrad_prompt.py" || echo "No TextGrad test processes found"
	@pkill -f "make test-textgrad" || echo "No make test-textgrad processes found"
	@echo "TextGrad test processes killed"

# Kill any running TextGrad+GEPA optimizer processes
kill-textgrad-gepa:
	@echo "Killing any running TextGrad+GEPA optimizer processes..."
	@pkill -f "textgrad_latest_gepa_optimizer.py" || echo "No TextGrad+GEPA optimizer processes found"
	@pkill -f "make optimize-textgrad-gepa" || echo "No make optimize-textgrad-gepa processes found"
	@echo "TextGrad+GEPA optimizer processes killed"

# Kill any running TextGrad+GEPA test processes
kill-test-textgrad-gepa:
	@echo "Killing any running TextGrad+GEPA test processes..."
	@pkill -f "test_latest_textgrad_gepa_prompt.py" || echo "No TextGrad+GEPA test processes found"
	@pkill -f "make test-textgrad-gepa" || echo "No make test-textgrad-gepa processes found"
	@echo "TextGrad+GEPA test processes killed"

# Clear DSPy cache
clear-cache:
	@echo "Clearing DSPy cache contents..."
	@if [ -d "cache" ]; then \
		find cache -mindepth 1 ! -name ".gitkeep" -delete; \
		echo "✓ Cleared cache/ contents (kept .gitkeep files)"; \
	else \
		echo "No cache/ directory found"; \
	fi
	@if [ -n "$$DSPY_CACHEDIR" ] && [ -d "$$DSPY_CACHEDIR" ]; then \
		find "$$DSPY_CACHEDIR" -mindepth 1 ! -name ".gitkeep" -delete; \
		echo "✓ Cleared custom cache directory contents: $$DSPY_CACHEDIR (kept .gitkeep files)"; \
	fi

# Show activation command
activate:
	@echo "To activate the virtual environment, run:"
	@echo "  source .venv/bin/activate"
	@echo ""
	@echo "Or use uv to run commands directly:"
	@echo "  uv run python your_script.py"