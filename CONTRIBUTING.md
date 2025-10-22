# Contributing to Breast Cancer Detection

Thank you for your interest in contributing to this project!

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/gpassoni/breast_cancer_detection.git
cd breast_cancer_detection
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

4. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Project Structure

- **`src/`**: Production code for model inference (clean, no training code)
- **`api/`**: FastAPI backend for serving predictions
- **`app/`**: Streamlit web interface
- **`training_and_experiments/`**: Training scripts and experiments
- **`scripts/`**: Original training utilities (legacy, kept for reference)

## Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings to all public functions and classes
- Keep functions focused and modular

## Making Changes

1. Create a new branch for your feature:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes and test them:
```bash
# Test the API
python -m api.main

# Test the web interface
streamlit run app/main.py
```

3. Commit your changes:
```bash
git add .
git commit -m "Description of your changes"
```

4. Push and create a pull request:
```bash
git push origin feature/your-feature-name
```

## Adding New Models

When adding new models to `src/models/`:
1. Create a new Python file (e.g., `my_model.py`)
2. Add the model class with clear documentation
3. Update `src/models/__init__.py` to export the model
4. Add prediction logic to `src/inference/predictor.py` if needed

## Testing

Currently, there is no automated test suite. When adding new features:
1. Manually test all affected endpoints
2. Verify the web interface still works
3. Check that models load correctly
4. Test with sample images if available

## Environment Variables

Never commit `.env` files or API keys. Always use `.env.example` as a template and add new variables there with placeholder values.

## Questions?

Feel free to open an issue for any questions or suggestions!
