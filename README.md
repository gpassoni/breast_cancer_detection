# Breast Cancer Detection using Thermal Imaging and Self-Supervised Learning  

## Introduction  
Breast cancer remains one of the leading causes of death among women worldwide. Early detection plays a crucial role in improving treatment success rates and survival outcomes. Traditional screening techniques, while effective, still face limitations in accessibility, invasiveness, and accuracy.  

In this project, we explore **thermal imaging** combined with **artificial intelligence (AI)** as an alternative, non-invasive approach for breast cancer detection. The main objective is to develop an AI-driven pipeline capable of assisting clinicians in identifying potential tumors at early stages, thus increasing the chances of successful treatment.  

## Technical Overview  
One of the major challenges in medical AI is the **lack of large, annotated datasets**. Building robust supervised models is particularly difficult when reliable labels are scarce.  

To overcome this limitation, we adopted a **Self-Supervised Learning (SSL)** strategy. By leveraging SSL, we can train models to learn meaningful representations of thermal breast images **without requiring labels**, making the approach scalable and data-efficient.  

The pipeline includes several key components:  
- **Segmentation**: Breast regions are segmented using **R2AU-Net** (implementation inspired by [LeeJunHyun/Image_Segmentation](https://github.com/LeeJunHyun/Image_Segmentation)) to remove noise and isolate the relevant areas.  
- **Representation Learning**: We used **VICReg** (a self-supervised method developed by [Meta AI](https://github.com/facebookresearch/vicreg)) to learn robust image embeddings from unlabeled thermal data.  
- **Downstream Tasks**: After pretraining, the learned embeddings are fine-tuned for classification tasks using limited labeled data.  

## Results & Comparison  
Despite the scarcity of annotated thermal datasets, our approach shows performance comparable to **state-of-the-art (SOTA)** studies in the domain. By leveraging SSL, we achieve competitive results while relying on **significantly fewer labeled samples**.  

All downstream evaluations are conducted on **publicly available datasets** such as **DMR-IR**, ensuring transparency and reproducibility of the benchmarking process.  

## Repository Structure (Production-Ready)

```
breast_cancer_detection/
├── src/                          # Production-ready model code
│   ├── models/                   # Neural network models
│   │   ├── segmentation.py      # R2AU-Net segmentation model
│   │   └── classification.py    # Classification head
│   ├── preprocessing/            # Data preprocessing utilities
│   │   └── preprocessing.py     # Image preprocessing functions
│   ├── inference/                # Model inference/prediction
│   │   └── predictor.py         # Main predictor class
│   └── config.py                 # Configuration management
│
├── api/                          # FastAPI backend
│   ├── main.py                   # FastAPI application
│   ├── schemas.py                # Pydantic models
│   └── routes/                   # API endpoints (future)
│
├── app/                          # Streamlit web interface
│   └── main.py                   # Streamlit application
│
├── training_and_experiments/     # Training scripts
│   ├── segmentation_model_trainer.py
│   ├── downstream_classification_trainer.py
│   └── ...                       # Other training scripts
│
├── scripts/                      # Original training utilities
│   ├── models/                   # Model architectures (for training)
│   ├── preprocessing/            # Data loaders (for training)
│   └── utils/                    # Utility functions
│
├── best_model_saved/             # Pre-trained model weights
│   ├── best_model.pth           # Segmentation model weights
│   └── config.json              # Model configuration
│
├── vicreg/                       # VICReg SSL checkpoints
│
├── exploration_and_testing/      # Exploratory notebooks
├── results/                      # Results and visualizations
├── .env.example                  # Environment variables template
├── requirements.txt              # Production dependencies
├── requirements-dev.txt          # Development dependencies
└── README.md                     # This file
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/gpassoni/breast_cancer_detection.git
cd breast_cancer_detection

# Install dependencies
pip install -r requirements.txt

# For development (includes training tools)
pip install -r requirements-dev.txt

# Copy and configure environment variables
cp .env.example .env
# Edit .env with your configuration
```

### 2. Configuration

Edit the `.env` file with your settings:

```bash
# Required for training
ROOT_DIR=/path/to/your/data
WANDB_API_KEY=your_wandb_api_key
WANDB_PROJECT=breast-cancer-detection
WANDB_ENTITY=your_wandb_entity

# Optional (has defaults)
API_HOST=0.0.0.0
API_PORT=8000
```

### 3. Running the Application

#### Start the API Server

```bash
cd breast_cancer_detection
python -m api.main
```

The API will be available at `http://localhost:8000`

#### Start the Streamlit Web Interface

```bash
cd breast_cancer_detection
streamlit run app/main.py
```

The web interface will open in your browser at `http://localhost:8501`

### 4. Using the API

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Segment an Image
```bash
curl -X POST "http://localhost:8000/predict/segment" \
  -F "file=@path/to/image.jpg"
```

### 5. Training Models

If you want to train your own models:

```bash
# Make sure your .env file is configured with data paths and wandb credentials

# Train segmentation model
cd training_and_experiments
python segmentation_model_trainer.py

# Train classification model
python downstream_classification_trainer.py
```

## API Endpoints

- `GET /` - Root endpoint
- `GET /health` - Health check and model status
- `POST /predict/segment` - Segment breast regions from thermal image
- `POST /predict/classify` - Classify image (coming soon)
- `POST /predict/full` - Run full prediction pipeline (coming soon)

## Features

### ✅ Implemented
- Production-ready model inference
- FastAPI backend with RESTful endpoints
- Streamlit web interface
- Environment-based configuration
- Clean, modular code structure

### 🚧 Coming Soon
- Classification model integration
- Full prediction pipeline
- Batch processing
- Docker deployment
- Model versioning

## Development

### Project Structure
- **`src/`**: Production code for model inference
- **`api/`**: FastAPI backend application
- **`app/`**: Streamlit web interface
- **`training_and_experiments/`**: Training scripts with wandb integration
- **`scripts/`**: Original training utilities and data loaders

### Environment Variables
All sensitive configuration (API keys, data paths) is managed through environment variables using `.env` files. Never commit `.env` files to the repository.

## Usage Notes

### Segmentation (R2AU-Net)
The segmentation model is production-ready and can be used via the API or web interface. Pre-trained weights are available in `best_model_saved/`.

### Self-Supervised Learning (VICReg)
Due to **high computational requirements**, SSL pretraining was executed on a cluster. The full VICReg source code is not included here.
- To reproduce SSL experiments, download the official VICReg implementation from [Meta AI's GitHub repository](https://github.com/facebookresearch/vicreg).
- The `/vicreg` folder contains **checkpoints** and configurations with the best parameters from our runs.

### Data Availability
⚠️ **Note**: The proprietary thermal images used for SSL pretraining and segmentation cannot be shared. However, all evaluation and testing have been carried out using **public datasets (e.g., DMR-IR)**, which are freely accessible for research purposes.

## Disclaimer

This is a research tool and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers.

## References

- R2AU-Net: [Image Segmentation by LeeJunHyun](https://github.com/LeeJunHyun/Image_Segmentation)
- VICReg: [Meta AI Research](https://github.com/facebookresearch/vicreg)
- DMR-IR Dataset: Public thermal breast imaging dataset

## License

See LICENSE file for details.

## Citation

If you use this work in your research, please cite:
```
[Citation information to be added]
```
