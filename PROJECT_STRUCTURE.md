# Project Structure Overview

This document provides a detailed overview of the reorganized project structure.

## Directory Layout

```
breast_cancer_detection/
│
├── 📦 PRODUCTION CODE
│   ├── src/                      # Core production inference code
│   │   ├── models/              # Neural network models
│   │   │   ├── segmentation.py # R2AU-Net for segmentation
│   │   │   └── classification.py # Classification head
│   │   ├── preprocessing/       # Image preprocessing
│   │   │   └── preprocessing.py # Utilities for image processing
│   │   ├── inference/           # Prediction pipeline
│   │   │   └── predictor.py    # Main predictor class
│   │   └── config.py            # Configuration management
│   │
│   ├── api/                      # FastAPI Backend
│   │   ├── main.py              # FastAPI application
│   │   ├── schemas.py           # Pydantic models
│   │   └── routes/              # API route handlers (future)
│   │
│   └── app/                      # Streamlit Frontend
│       └── main.py              # Web interface
│
├── 🔬 TRAINING & RESEARCH
│   ├── training_and_experiments/ # Training scripts
│   │   ├── segmentation_model_trainer.py
│   │   ├── downstream_classification_trainer.py
│   │   ├── segmentation_cv.py
│   │   ├── segmentation_augmentation_study.py
│   │   ├── downstream_classification_datastudy.py
│   │   └── segmentation_sweep_trainer.py
│   │
│   ├── scripts/                  # Legacy training utilities
│   │   ├── models/              # Training model implementations
│   │   ├── preprocessing/       # Training data loaders
│   │   └── utils/               # Utility functions
│   │
│   ├── vicreg/                   # VICReg SSL checkpoints
│   │   └── output_ssl_cluster*/ # Pre-trained SSL models
│   │
│   └── exploration_and_testing/  # Jupyter notebooks for EDA
│
├── 💾 MODELS & DATA
│   ├── best_model_saved/         # Pre-trained model weights
│   │   ├── best_model.pth       # Segmentation model
│   │   └── config.json          # Model configuration
│   │
│   └── results/                  # Training results
│
├── 🐳 DEPLOYMENT
│   ├── Dockerfile               # API container
│   ├── Dockerfile.streamlit     # Web interface container
│   ├── docker-compose.yml       # Multi-service orchestration
│   ├── run_api.sh              # Helper script to run API
│   └── run_app.sh              # Helper script to run web app
│
├── 📝 DOCUMENTATION
│   ├── README.md               # Main documentation
│   ├── CONTRIBUTING.md         # Contribution guide
│   ├── DEPLOYMENT.md           # Deployment instructions
│   ├── PROJECT_STRUCTURE.md    # This file
│   └── readme_old.md          # Original README (backup)
│
├── ⚙️ CONFIGURATION
│   ├── .env.example            # Environment variables template
│   ├── .gitignore             # Git ignore rules
│   ├── requirements.txt       # Production dependencies
│   ├── requirements-dev.txt   # Development dependencies
│   ├── environment.yml        # Conda environment (legacy)
│   ├── sweep.yaml            # Wandb sweep config
│   └── sweep_multiclass.yaml # Multiclass sweep config
│
└── 📊 ANALYSIS
    ├── results_segmentation.ipynb
    ├── results_downstream_classification.ipynb
    └── results_augmentation_ablation_study.ipynb
```

## Key Components

### 🎯 Production Stack

**src/** - Clean, production-ready code
- No training dependencies
- Focused on inference only
- Well-documented and tested
- Environment-agnostic

**api/** - RESTful API server
- FastAPI framework
- Async endpoints
- Automatic documentation (Swagger/OpenAPI)
- Health checks and monitoring

**app/** - User interface
- Streamlit web framework
- Interactive visualization
- Real-time predictions
- User-friendly design

### 🔬 Research & Development

**training_and_experiments/** - Training scripts
- Wandb integration (via .env)
- Cross-validation
- Hyperparameter tuning
- Ablation studies

**scripts/** - Legacy training code
- Original implementations
- Data loaders for training
- Model architectures with training logic
- Kept for reproducibility

### 📦 Pre-trained Models

**best_model_saved/** - Production model
- R2AU-Net segmentation model
- Configuration JSON
- Best performing weights

**vicreg/** - SSL checkpoints
- Self-supervised learning models
- Multiple training runs
- Best parameters documented

## Data Flow

```
┌─────────────┐
│   User      │
│  (Browser)  │
└─────┬───────┘
      │
      ▼
┌─────────────────┐
│   Streamlit     │
│   Web App       │
│   (app/main.py) │
└─────┬───────────┘
      │ HTTP POST
      ▼
┌─────────────────┐
│   FastAPI       │
│   Backend       │
│   (api/main.py) │
└─────┬───────────┘
      │
      ▼
┌─────────────────────────┐
│   BreastCancerPredictor │
│   (src/inference/)      │
└─────┬───────────────────┘
      │
      ├─────────────┬────────────────┐
      ▼             ▼                ▼
┌───────────┐ ┌──────────┐  ┌──────────────┐
│  R2AU-Net │ │  Preproc │  │ Config       │
│  Model    │ │  Utils   │  │ Management   │
└───────────┘ └──────────┘  └──────────────┘
```

## Environment Variables

All configuration is managed through `.env` file:

```bash
# Production (required)
API_HOST=0.0.0.0
API_PORT=8000

# Training (optional)
ROOT_DIR=/path/to/data
WANDB_API_KEY=your_key
WANDB_PROJECT=project_name
WANDB_ENTITY=entity_name
```

## Development Workflow

1. **Research**: Experiment in `training_and_experiments/`
2. **Train**: Use training scripts with wandb tracking
3. **Export**: Save best model to `best_model_saved/`
4. **Integrate**: Update `src/inference/` if needed
5. **Test**: Run API and web app locally
6. **Deploy**: Use Docker for production

## Testing Strategy

### Manual Testing
- API endpoints via curl or Postman
- Web interface visual testing
- Model loading verification

### Future Automated Testing
- Unit tests for `src/` modules
- Integration tests for API
- E2E tests for web interface

## Deployment Options

1. **Local Development**: Use helper scripts
   ```bash
   ./run_api.sh    # Terminal 1
   ./run_app.sh    # Terminal 2
   ```

2. **Docker (Recommended)**: Single command deployment
   ```bash
   docker-compose up -d
   ```

3. **Cloud**: Deploy to AWS, GCP, or Azure
   - See DEPLOYMENT.md for details

## Code Organization Principles

✅ **Separation of Concerns**
- Production code in `src/`
- Training code in `training_and_experiments/`
- Deployment config in Docker files

✅ **Environment-based Configuration**
- No hardcoded secrets
- All config via `.env` file
- Template in `.env.example`

✅ **Minimal Dependencies**
- Production: only inference deps
- Development: adds training tools
- Clear separation in requirements files

✅ **Documentation First**
- README for users
- CONTRIBUTING for developers
- DEPLOYMENT for ops
- Inline docstrings for code

## Migration from Old Structure

### Before (Old)
```
scripts/        # Mixed training and inference
├── models/     # Training + inference
├── preprocessing/
└── utils/

training_and_experiments/  # Hardcoded configs
```

### After (New)
```
src/           # Clean inference only
├── models/
├── preprocessing/
└── inference/

api/           # Production API
app/           # Production UI
training_and_experiments/  # Uses .env config
```

## Key Improvements

1. ✅ **Production-ready structure** - Clear separation
2. ✅ **Environment-based config** - No hardcoded values
3. ✅ **Docker support** - Easy deployment
4. ✅ **API and Web UI** - User-friendly interfaces
5. ✅ **Clean code** - Modular and documented
6. ✅ **Helper scripts** - Developer convenience

## Next Steps

For future development:
- [ ] Add automated tests
- [ ] Implement classification model
- [ ] Add model versioning
- [ ] Set up CI/CD pipeline
- [ ] Add monitoring and logging
- [ ] Create performance benchmarks
