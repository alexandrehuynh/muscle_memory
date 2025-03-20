# Muscle Memory

## Core Value Proposition
A B2B service package for fitness companies and apps that integrates advanced AI and computer vision technology to provide comprehensive movement analysis, personalized coaching, and business intelligence - all accessible through a smartphone.

## Project Structure
- `api/`: FastAPI endpoints and routers
- `models/`: Core analysis models
  - `movement/`: Movement analysis using MediaPipe
  - `ai_agent/`: AI coaching agent
- `utils/`: Utility functions and helpers
- `tests/`: Unit and integration tests
- `docs/`: Documentation

## Setup

### Installation
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt