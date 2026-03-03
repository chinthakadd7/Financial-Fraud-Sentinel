from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any
import uuid
import time

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.schemas import (
    TransactionRequest,
    PredictionResponse,
    HealthResponse,
    HealthStatus,
    RiskLevelEnum,
    FeatureContribution,
    ErrorResponse
)
from src.models.predictor import get_predictor
from src.config import settings
from src.utils.logger import api_logger as logger, RequestLogger

# Application startup/shutdown lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):

    # Startup
    logger.info(f"Starting {settings.APP_NAME} v{settings.VERSION}")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Model path: {settings.MODEL_PATH}")
    
    # Warm up predictor (initialize H2O and load model)
    try:
        predictor = get_predictor()
        predictor.warmup()
        logger.info("Model warmup completed successfully")
    except Exception as e:
        logger.error(f"Model warmup failed: {e}. API will initialize on first request.")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application")
    try:
        predictor = get_predictor()
        predictor.shutdown()
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    description="AI-powered fraud detection API with explainable AI (SHAP) features",
    version=settings.VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)


# CORS middleware for dashboard integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request ID and logging middleware
@app.middleware("http")
async def request_middleware(request: Request, call_next):

    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    request.state.start_time = time.time()
    
    # Log incoming request
    logger.info(
        f"{request.method} {request.url.path}",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "client_host": request.client.host if request.client else None
        }
    )
    
    # Process request
    try:
        response = await call_next(request)
    except Exception as e:
        logger.error(f"Unhandled error: {e}", exc_info=True, extra={"request_id": request_id})
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="Internal Server Error",
                message="An unexpected error occurred",
                request_id=request_id,
                timestamp=datetime.utcnow()
            ).model_dump(mode="json")
        )
    
    # Calculate request duration
    duration_ms = (time.time() - request.state.start_time) * 1000
    
    # Add request ID to response headers
    response.headers["X-Request-ID"] = request_id
    
    # Log response
    logger.info(
        f"Response {response.status_code}",
        extra={
            "request_id": request_id,
            "status_code": response.status_code,
            "duration_ms": round(duration_ms, 2)
        }
    )
    
    return response


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
 
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail if isinstance(exc.detail, str) else "HTTP Error",
            message=exc.detail if isinstance(exc.detail, str) else str(exc.detail),
            request_id=getattr(request.state, "request_id", None),
            timestamp=datetime.utcnow()
        ).model_dump(mode="json")
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
 
    logger.error(
        f"Unhandled exception: {exc}",
        exc_info=True,
        extra={"request_id": getattr(request.state, "request_id", None)}
    )
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error",
            message="An unexpected error occurred" if settings.is_production else str(exc),
            request_id=getattr(request.state, "request_id", None),
            timestamp=datetime.utcnow()
        ).model_dump(mode="json")
    )

 
@app.get(
    "/",
    tags=["Info"],
    summary="API Information",
    response_model=Dict[str, str]
)
async def root():

    return {
        "name": settings.APP_NAME,
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict"
    }


@app.get(
    "/health",
    tags=["Health"],
    summary="Health Check",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK
)
async def health_check():
 
    try:
        predictor = get_predictor()
        health_info = predictor.health_check()
        
        # Determine overall health status
        if health_info["model_loaded"] and health_info["h2o_connected"]:
            overall_status = HealthStatus.HEALTHY
            status_code = status.HTTP_200_OK
        elif health_info["h2o_initialized"]:
            overall_status = HealthStatus.DEGRADED
            status_code = status.HTTP_200_OK
        else:
            overall_status = HealthStatus.UNHEALTHY
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        
        response = HealthResponse(
            status=overall_status,
            timestamp=datetime.utcnow(),
            version=settings.VERSION,
            model_loaded=health_info["model_loaded"],
            h2o_connected=health_info.get("h2o_connected", False),
            uptime_seconds=health_info.get("uptime_seconds", 0.0),
            details={
                "h2o_version": health_info.get("h2o_version"),
                "model_name": settings.MODEL_NAME,
                "model_path": str(health_info.get("model_path", "")),
                "environment": settings.ENVIRONMENT
            }
        )
        
        if overall_status == HealthStatus.UNHEALTHY:
            return JSONResponse(
                status_code=status_code,
                content=response.model_dump(mode="json")
            )
        
        return response
        
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=HealthResponse(
                status=HealthStatus.UNHEALTHY,
                timestamp=datetime.utcnow(),
                version=settings.VERSION,
                model_loaded=False,
                h2o_connected=False,
                uptime_seconds=0.0,
                details={"error": str(e)}
            ).model_dump(mode="json")
        )


@app.post(
    "/predict",
    tags=["Prediction"],
    summary="Predict Fraud",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Successful prediction"},
        422: {"description": "Validation error", "model": ErrorResponse},
        500: {"description": "Model prediction failed", "model": ErrorResponse},
        503: {"description": "Service unavailable", "model": ErrorResponse}
    }
)
async def predict_fraud(transaction: TransactionRequest, request: Request):
 
    request_id = request.state.request_id
    
    try:
        with RequestLogger(logger, request_id) as log:
            log.info("Processing fraud prediction request")
            
            # Get predictor and make prediction
            predictor = get_predictor()
            result = predictor.predict_transaction(transaction.dict())
            
            # Classify risk level
            fraud_prob = result['fraud_probability']
            if fraud_prob >= 0.7:
                risk_level = RiskLevelEnum.HIGH
            elif fraud_prob >= 0.3:
                risk_level = RiskLevelEnum.MEDIUM
            else:
                risk_level = RiskLevelEnum.LOW
            
            # Format top features
            top_features = [
                FeatureContribution(feature=feat, contribution=contrib)
                for feat, contrib in result['top_features'].items()
            ]
            
            # Build response
            response = PredictionResponse(
                request_id=request_id,
                fraud_probability=fraud_prob,
                prediction=result['prediction'],
                risk_level=risk_level,
                top_features=top_features,
                timestamp=datetime.utcnow(),
                model_version=settings.MODEL_NAME
            )
            
            log.info(
                f"Prediction complete: fraud={result['prediction']}, prob={fraud_prob:.3f}, risk={risk_level}",
                extra={
                    "fraud_detected": result['prediction'] == 1,
                    "fraud_probability": fraud_prob,
                    "risk_level": risk_level.value
                }
            )
            
            return response
            
    except ValueError as e:
        # Validation errors (422)
        logger.error(
            f"Validation error: {e}",
            extra={"request_id": request_id}
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Validation error: {str(e)}"
        )
        
    except RuntimeError as e:
        # Model failures (500)
        logger.error(
            f"Model prediction failed: {e}",
            exc_info=True,
            extra={"request_id": request_id}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model prediction failed. Please try again."
        )
        
    except Exception as e:
        # Unexpected errors (500)
        logger.error(
            f"Unexpected error during prediction: {e}",
            exc_info=True,
            extra={"request_id": request_id}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred"
        )


if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting API server on {settings.API_HOST}:{settings.API_PORT}")
    
    uvicorn.run(
        "src.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True
    )
