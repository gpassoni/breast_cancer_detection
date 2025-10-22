# Deployment Guide

This guide explains how to deploy the Breast Cancer Detection system in different environments.

## Prerequisites

- Python 3.11+
- Docker and Docker Compose (for containerized deployment)
- Pre-trained model weights in `best_model_saved/`

## Local Development

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your settings
```

### 2. Run Services

**Terminal 1 - API Server:**
```bash
./run_api.sh
# Or manually: python -m api.main
```

**Terminal 2 - Web Interface:**
```bash
./run_app.sh
# Or manually: streamlit run app/main.py
```

### 3. Access Services

- API: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Web Interface: http://localhost:8501

## Docker Deployment (Recommended for Production)

### Quick Start

```bash
# Build and start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Configuration

Edit `docker-compose.yml` to customize:
- Ports
- Environment variables
- Volume mounts

### Accessing Services

- API: http://localhost:8000
- Web Interface: http://localhost:8501

## Cloud Deployment

### AWS Elastic Beanstalk

1. Install EB CLI:
```bash
pip install awsebcli
```

2. Initialize:
```bash
eb init -p docker breast-cancer-detection
```

3. Deploy:
```bash
eb create production
eb open
```

### Google Cloud Run

1. Build container:
```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/breast-cancer-api
```

2. Deploy:
```bash
gcloud run deploy breast-cancer-api \
  --image gcr.io/PROJECT_ID/breast-cancer-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Azure Container Instances

1. Build and push:
```bash
az acr build --registry REGISTRY_NAME \
  --image breast-cancer-api:latest .
```

2. Deploy:
```bash
az container create \
  --resource-group RESOURCE_GROUP \
  --name breast-cancer-api \
  --image REGISTRY_NAME.azurecr.io/breast-cancer-api:latest \
  --ports 8000
```

## Environment Variables

Required for production:

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Model Configuration (optional, has defaults)
SEGMENTATION_MODEL_PATH=best_model_saved/best_model.pth
SEGMENTATION_CONFIG_PATH=best_model_saved/config.json
```

Optional for training:

```bash
# Data paths
ROOT_DIR=/path/to/data

# Weights & Biases
WANDB_API_KEY=your_key
WANDB_PROJECT=project_name
WANDB_ENTITY=entity_name
```

## Health Checks

Check if services are running:

```bash
# API health check
curl http://localhost:8000/health

# Expected response:
# {"status":"healthy","message":"API is operational","models_loaded":{...}}
```

## Troubleshooting

### API won't start
- Check if port 8000 is available
- Verify model weights exist in `best_model_saved/`
- Check logs: `docker-compose logs api`

### Streamlit can't connect to API
- Ensure API is running first
- Check API_HOST and API_PORT in .env
- Verify network connectivity (for Docker, containers should be on same network)

### Model loading errors
- Verify model files exist: `best_model_saved/best_model.pth`
- Check config.json format
- Ensure PyTorch is installed: `pip list | grep torch`

## Security Considerations

1. **Never commit .env files** - Use .env.example as template
2. **Use HTTPS in production** - Configure SSL/TLS certificates
3. **Restrict API access** - Add authentication if needed
4. **Keep dependencies updated** - Regularly update requirements.txt
5. **Scan for vulnerabilities** - Use tools like `safety` or `bandit`

## Performance Tuning

### API Optimization
- Use Gunicorn with multiple workers:
  ```bash
  gunicorn api.main:app -w 4 -k uvicorn.workers.UvicornWorker
  ```
- Enable model caching
- Use GPU for inference if available

### Resource Requirements
- **Minimum**: 2 CPU cores, 4GB RAM
- **Recommended**: 4 CPU cores, 8GB RAM
- **With GPU**: CUDA-capable GPU with 4GB+ VRAM

## Monitoring

Consider adding:
- Prometheus for metrics
- Grafana for visualization
- Sentry for error tracking
- Application logs with structured logging

## Backup and Recovery

Important files to backup:
- `best_model_saved/` - Model weights and config
- `.env` - Environment configuration (secure storage)
- Training data (if available)

## Scaling

For high-traffic scenarios:
- Use load balancer (Nginx, HAProxy)
- Deploy multiple API instances
- Consider Kubernetes for orchestration
- Use caching layer (Redis) for frequent predictions
