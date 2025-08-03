# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
```bash
# Development mode with auto-reload
python main.py

# Or using uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Docker deployment
docker-compose up --build
```

### Testing
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=.
```

### Fine-tuning
```bash
# Run fine-tuning script
./scripts/run_fine_tune.sh

# Or directly
python scripts/fine_tuning/fine_tune.py
```

### Dependencies
```bash
# Install requirements
pip install -r requirements.txt
```

## Architecture Overview

ShachikuAI is a FastAPI-based excuse generation service using local LLMs with fine-tuning capabilities.

### Core Components

1. **API Layer** (`api/v1/`)
   - `excuse_router.py`: FastAPI router handling `/v1/excuse/generate` endpoint
   - Uses dependency injection for service layer integration

2. **Service Layer** (`service/excuse_generation/`)
   - `excuse_service.py`: Core business logic for excuse generation
   - Handles prompt creation, text formatting, confidence scoring, and fallback responses
   - Integrates with model client for LLM interaction

3. **Client Layer** (`client/llm/`)
   - `model_client.py`: Abstracts LLM interactions using Transformers/HuggingFace
   - Supports both local and remote model loading
   - Handles device management (CPU/CUDA) and generation configuration

4. **Models** (`models/`)
   - `request_models.py`: Pydantic models for API request/response validation

### Data Flow

1. API receives excuse generation request via `/v1/excuse/generate`
2. Router delegates to `ExcuseService` 
3. Service creates Japanese prompt and calls `ModelClient`
4. Model client handles LLM inference using Transformers pipeline
5. Service formats response and calculates confidence score
6. API returns structured JSON response

### Configuration

- Environment variables loaded via `.env` file
- Model selection: `MODEL_NAME` (defaults to microsoft/DialoGPT-medium)
- Model storage: `MODEL_PATH` for local fine-tuned models
- API settings: `API_HOST`, `API_PORT`

### Fine-tuning Pipeline

- Training data: JSONL format in `data/training/excuses.jsonl`
- Fine-tuning script: `scripts/fine_tuning/fine_tune.py`
- Output models saved to `data/models/fine_tuned/`
- Configuration in `config/llm/fine_tune_config.py`

### Key Features

- Japanese language excuse generation with polite expressions
- Fallback responses for error scenarios
- Confidence scoring based on text characteristics
- Docker containerization support
- Health check endpoints for monitoring