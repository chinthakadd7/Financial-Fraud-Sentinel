# 🛡️ Financial Fraud Sentinel

**AI-Powered Financial Fraud Detection System with Explainable AI**

A production-ready fraud detection platform leveraging H2O AutoML, FastAPI, and H2O Wave for real-time transaction analysis with explainable AI capabilities.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![H2O](https://img.shields.io/badge/H2O-3.44.0.3-orange.svg)](https://h2o.ai/)
[![Docker](https://img.shields.io/badge/Docker-ready-blue.svg)](https://www.docker.com/)

---

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [Configuration](#-configuration)
- [Testing](#-testing)
- [Deployment](#-deployment)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)

---

## ✨ Features

### Core Capabilities
- **Real-Time Fraud Detection**: Analyze transactions in milliseconds using H2O AutoML XGBoost
- **Explainable AI**: SHAP values for feature importance and prediction transparency
- **REST API**: Production-grade FastAPI with automatic OpenAPI documentation
- **Interactive Dashboard**: Modern H2O Wave interface for business users
- **API-First Architecture**: Microservices design for scalability and flexibility

### Technical Features
- ✅ Request validation with Pydantic schemas
- ✅ Structured JSON logging for cloud-native monitoring
- ✅ Health checks and readiness probes for orchestration
- ✅ Docker containerization with multi-stage builds
- ✅ Environment-based configuration (12-factor app)
- ✅ Comprehensive test suite (unit + integration)
- ✅ Request tracing with unique IDs
- ✅ CORS support for cross-origin requests

---

## 🏗️ Architecture

```
┌─────────────────┐         HTTP          ┌─────────────────┐
│                 │ ◄──────────────────► │                 │
│   Dashboard     │    (prediction)      │    FastAPI      │
│   (H2O Wave)    │                      │      API        │
│   Port: 10101   │                      │   Port: 8000    │
└─────────────────┘                      └────────┬────────┘
                                                   │
                                                   │ calls
                                                   ▼
                                          ┌─────────────────┐
                                          │   H2O Model     │
                                          │   (XGBoost)     │
                                          │   + SHAP        │
                                          └─────────────────┘
```

**Component Communication:**
- **Dashboard → API**: Async HTTP calls via httpx
- **API → Model**: Direct Python integration with H2O predictor
- **Model → H2O**: MOJO model loading and prediction

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed design documentation.

---

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```powershell
# Clone repository
git clone <repository-url>
cd financial-fraud-sentinel

# Start all services
docker-compose up -d

# Verify services are running
docker-compose ps

# Check API health
curl http://localhost:8000/health

# Access dashboard
# Open browser: http://localhost:10101/fraud
```

### Option 2: Local Development

```powershell
# Install dependencies
pip install -r requirements.txt

# Copy environment configuration
cp .env.example .env

# Start API server (Terminal 1)
python -m uvicorn src.api.main:app --reload --port 8000

# Start dashboard (Terminal 2)
wave run src.dashboard.app

# Access services
# API: http://localhost:8000/docs
# Dashboard: http://localhost:10101/fraud
```

---

## 📦 Installation

### Prerequisites

- **Python**: 3.10 or higher
- **Java**: JDK 17+ (required for H2O)
- **Memory**: Minimum 4GB RAM (2GB for H2O, 1GB for API, 1GB for dashboard)
- **Docker**: Optional but recommended for production

### System Requirements

| Component | CPU | Memory | Disk |
|-----------|-----|--------|------|
| API + H2O | 1-2 cores | 3GB | 500MB |
| Dashboard | 0.5-1 core | 1GB | 100MB |
| **Total** | **2-3 cores** | **4GB** | **600MB** |

### Step-by-Step Installation

1. **Clone Repository**
   ```powershell
   git clone <repository-url>
   cd financial-fraud-sentinel
   ```

2. **Create Virtual Environment**
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   # source venv/bin/activate  # Linux/Mac
   ```

3. **Install Dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Configure Environment**
   ```powershell
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Verify Model Artifacts**
   ```powershell
   # Ensure model file exists
   ls models_artifacts/XGBoost_1_AutoML_1_20260209_165338.zip
   ```

---

## 💻 Usage

### API Usage

#### Test with curl
```bash
# Health check
curl http://localhost:8000/health

# Fraud prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "TransactionAmt": 250.75,
    "card1": 12345,
    "DeviceType": "mobile"
  }'
```

#### Test with Python
```python
import httpx

async def predict_fraud():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/predict",
            json={
                "TransactionAmt": 250.75,
                "card1": 12345,
                "DeviceType": "mobile"
            }
        )
        return response.json()
```

### Dashboard Usage

1. Open browser: `http://localhost:10101/fraud`
2. Enter transaction details:
   - **Transaction Amount**: Dollar value (e.g., 150.00)
   - **Card ID**: Card identifier (e.g., 12345)
3. Click **"Analyze Transaction"**
4. Review results:
   - Fraud probability percentage
   - Risk level (LOW/MEDIUM/HIGH)
   - Top contributing features (SHAP)
   - Recommendations

---

## 📚 API Documentation

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API information |
| `GET` | `/health` | Health check |
| `POST` | `/predict` | Fraud prediction |
| `GET` | `/docs` | Swagger UI |
| `GET` | `/redoc` | ReDoc UI |

### Example Response

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "fraud_probability": 0.87,
  "prediction": 1,
  "risk_level": "HIGH",
  "top_features": [
    {"feature": "TransactionAmt", "contribution": 0.45},
    {"feature": "card1", "contribution": 0.23},
    {"feature": "D1", "contribution": 0.12}
  ],
  "timestamp": "2026-02-27T10:30:00",
  "model_version": "XGBoost_1_AutoML_1_20260209_165338"
}
```

For complete API documentation, see [docs/API.md](docs/API.md) or visit `/docs` when API is running.

---

## ⚙️ Configuration

Configuration is managed via environment variables (12-factor app methodology).

### Key Configuration Options

```bash
# Application
APP_NAME=Financial Fraud Sentinel
ENVIRONMENT=development  # development, staging, production

# API
API_HOST=0.0.0.0
API_PORT=8000

# Model
MODEL_PATH=models_artifacts/XGBoost_1_AutoML_1_20260209_165338.zip

# H2O
H2O_MEMORY_GB=2
H2O_PORT=54321

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
```

See [.env.example](.env.example) for all available options.

---

## 🧪 Testing

### Run All Tests
```powershell
pytest tests/ -v
```

### Run Unit Tests Only (Fast)
```powershell
pytest tests/test_prediction.py tests/test_api.py -v
```

### Run Integration Tests (Requires Running API)
```powershell
# Start API first
python -m uvicorn src.api.main:app &

# Run integration tests
pytest tests/test_integration.py -v -m integration
```

### Run with Coverage
```powershell
pytest --cov=src --cov-report=html
# Open htmlcov/index.html
```

### Test Markers
- `unit`: Fast unit tests (no external dependencies)
- `integration`: Integration tests (requires API)
- `slow`: Slow tests (H2O initialization)

---

## 🚢 Deployment

### Docker Deployment

```powershell
# Build images
docker-compose build

# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Production Considerations

1. **Environment Variables**: Use secrets management (AWS Secrets Manager, Azure Key Vault)
2. **Resource Limits**: Configure memory/CPU limits in docker-compose.yml
3. **Health Checks**: Configure load balancer to use `/health` endpoint
4. **Logging**: Forward logs to centralized logging (ELK, CloudWatch)
5. **Monitoring**: Add Prometheus metrics and Grafana dashboards
6. **Scaling**: Use Kubernetes for horizontal scaling

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for detailed deployment guide.

---

## 📁 Project Structure

```
financial-fraud-sentinel/
├── src/
│   ├── api/
│   │   ├── main.py              # FastAPI application
│   │   └── schemas.py           # Pydantic models
│   ├── dashboard/
│   │   └── app.py               # H2O Wave dashboard
│   ├── models/
│   │   └── predictor.py         # Model prediction logic
│   ├── data_pipeline/
│   │   └── preprocess.py        # Data preprocessing
│   ├── utils/
│   │   └── logger.py            # Structured logging
│   └── config.py                # Configuration management
├── models_artifacts/
│   └── XGBoost_1_AutoML_*.zip   # Trained model
├── tests/
│   ├── test_prediction.py       # Model tests
│   ├── test_api.py              # API tests
│   └── test_integration.py      # Integration tests
├── notebooks/
│   ├── data_preparation.ipynb   # Data preprocessing
│   └── model_training.ipynb     # Model training
├── data/
│   ├── raw/                     # Raw data files
│   └── processed/               # Processed datasets
├── docs/                        # Documentation
├── Dockerfile.api               # API container
├── Dockerfile.dashboard         # Dashboard container
├── docker-compose.yml           # Docker orchestration
├── requirements.txt             # Python dependencies
├── pytest.ini                   # Test configuration
├── .env.example                 # Environment template
└── README.md                    # This file
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```powershell
# Install dev dependencies
pip install -r requirements.txt
pip install pytest-cov black flake8 mypy

# Run linters
black src/ tests/
flake8 src/ tests/

# Run type checking
mypy src/
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **H2O.ai**: For the AutoML framework and MOJO deployment
- **FastAPI**: For the excellent API framework
- **H2O Wave**: For the modern dashboard framework
- **SHAP**: For explainable AI capabilities

---

## 📞 Support

For issues, questions, or contributions:
- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Documentation**: [docs/](docs/)
- **API Docs**: http://localhost:8000/docs (when running)

---

**Built with ❤️ for secure financial transactions**
