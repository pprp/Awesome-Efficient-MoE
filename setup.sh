#!/bin/bash

echo "🚀 Setting up Awesome-Efficient-MoE environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "⚡ Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "🔧 Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing Python dependencies..."
pip install -r scripts/requirements.txt

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  Warning: .env file not found!"
    echo "Please create .env file with your API keys:"
    echo "MINERU_API_KEY=your_mineru_key_here"
    echo "OPENAI_API_KEY=your_openai_key_here"
    echo "OPENAI_BASE_URL=your_openai_base_url_here"
else
    echo "✅ .env file found"
fi

echo "🎉 Setup completed!"
echo ""
echo "To test the image extraction functionality:"
echo "source venv/bin/activate"
echo "cd scripts"
echo "python test_image_extraction.py"
echo ""
echo "To run the main paper update script:"
echo "python update_papers.py --test  # for testing"
echo "python update_papers.py         # for actual execution"