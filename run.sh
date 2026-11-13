#!/usr/bin/env bash
# ==============================================================================
# 📚 BookNest AI — 1-Click Production Server Runner
# ==============================================================================
set -e

PORT=${PORT:-8000}
HOST=${HOST:-0.0.0.0}

echo "================================================="
echo "  🚀 Starting BookNest AI — Visual Book Sanctuary"
echo "================================================="

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is required but not installed."
    exit 1
fi

# Ensure data directory exists
mkdir -p data static/avatars

# Check dependencies
echo "📦 Verifying Python dependencies..."
python3 -c "import fastapi, uvicorn, PIL" 2>/dev/null || {
    echo "⚠️  Missing dependencies. Installing requirements..."
    pip3 install -r requirements.txt
}

echo "✨ Starting Uvicorn ASGI Server on http://${HOST}:${PORT}..."
exec python3 -m uvicorn main:app --host "${HOST}" --port "${PORT}" --workers 1
