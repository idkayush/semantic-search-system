#!/bin/bash

# Setup script for Semantic Search System
# This script sets up the virtual environment and installs dependencies

echo "========================================="
echo " Semantic Search System - Setup"
echo "========================================="

# Check Python version
echo ""
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Check if Python 3.11+ is available
required_version="3.11"
if [[ $(echo -e "$python_version\n$required_version" | sort -V | head -n1) != "$required_version" ]]; then
    echo "❌ Error: Python 3.11 or higher is required"
    echo "Current version: $python_version"
    exit 1
fi

echo "✓ Python version OK"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists. Skipping creation."
else
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing dependencies..."
echo "This may take several minutes..."
pip install -r requirements.txt

echo ""
echo "========================================="
echo " Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Run the pipeline to prepare data:"
echo "   python pipeline.py"
echo ""
echo "3. Start the API server:"
echo "   uvicorn api:app --reload"
echo ""
echo "4. Test the API:"
echo "   python test_api.py"
echo ""
echo "Or use Docker:"
echo "   docker-compose up --build"
echo ""
echo "========================================="
