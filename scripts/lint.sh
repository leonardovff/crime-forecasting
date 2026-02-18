#!/usr/bin/env bash
set -euo pipefail

echo "Formatting with isort and black..."
isort . --skip .pixi
black . --exclude '\.pixi'

echo "Running ruff..."
ruff check . --exclude .pixi

echo "Type checking with mypy..."
mypy . --exclude '\.pixi'

echo "Linting complete."

