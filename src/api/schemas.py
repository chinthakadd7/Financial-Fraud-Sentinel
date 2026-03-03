from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict


class DeviceTypeEnum(str, Enum):
    MOBILE = "mobile"
    DESKTOP = "desktop"
    TABLET = "tablet"
    UNKNOWN = "unknown"


class RiskLevelEnum(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class TransactionRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "TransactionAmt": 200075.50,
                "card1": 19945,
                "addr1": 19999,
                "DeviceType": "mobile",
                "C1": 3,
                "D1": 25.0,
                "card2": 150,
                "card3": 185,
                "card4": "visa",
                "card5": 226,
                "card6": "debit",
                "addr2": 87,
                "P_emaildomain": "gmail.com",
                "R_emaildomain": "hotmail.com",
                "M1": "T",
                "M2": "F",
                "M3": "T"
            }
        }
    )
    
    # Required fields
    TransactionAmt: float = Field(
        ..., 
        description="Transaction amount in USD",
        gt=0,
        examples=[200075.50]
    )
    card1: int = Field(
        ..., 
        description="Card identifier (primary card number)",
        examples=[19945]
    )
    
    # Optional fields with defaults
    addr1: Optional[int] = Field(
        None, 
        description="Billing address identifier",
        examples=[19999]
    )
    DeviceType: Optional[DeviceTypeEnum] = Field(
        DeviceTypeEnum.UNKNOWN,
        description="Device type used for transaction"
    )
    C1: Optional[int] = Field(
        None, 
        description="Count feature C1",
        examples=[3]
    )
    D1: Optional[float] = Field(
        None, 
        description="Timedelta feature D1 (days)",
        examples=[25.0]
    )
    card2: Optional[int] = Field(
        None,
        description="Card identifier (secondary)",
        examples=[150]
    )
    card3: Optional[int] = Field(
        None,
        description="Card identifier (tertiary)",
        examples=[185]
    )
    card4: Optional[str] = Field(
        None,
        description="Card network (visa, mastercard, etc.)",
        examples=["visa"]
    )
    card5: Optional[int] = Field(
        None,
        description="Card identifier (type 5)",
        examples=[226]
    )
    card6: Optional[str] = Field(
        None,
        description="Card type (debit, credit)",
        examples=["debit"]
    )
    addr2: Optional[int] = Field(
        None,
        description="Shipping address identifier",
        examples=[87]
    )
    P_emaildomain: Optional[str] = Field(
        None,
        description="Purchaser email domain",
        examples=["gmail.com"]
    )
    R_emaildomain: Optional[str] = Field(
        None,
        description="Recipient email domain",
        examples=["hotmail.com"]
    )
    M1: Optional[str] = Field(
        None,
        description="Match feature M1 (T/F)",
        examples=["T"]
    )
    M2: Optional[str] = Field(
        None,
        description="Match feature M2 (T/F)",
        examples=["F"]
    )
    M3: Optional[str] = Field(
        None,
        description="Match feature M3 (T/F)",
        examples=["T"]
    )
    
    @field_validator("TransactionAmt")
    @classmethod
    def validate_transaction_amount(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Transaction amount must be positive")
        if v > 1_000_000:
            raise ValueError("Transaction amount exceeds maximum ($1,000,000)")
        return v
    
    @field_validator("card1")
    @classmethod
    def validate_card1(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Card identifier must be non-negative")
        return v


class FeatureContribution(BaseModel):
    feature: str = Field(..., description="Feature name")
    contribution: float = Field(..., description="SHAP contribution value")


class PredictionResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
        }
    )
    
    request_id: str = Field(
        ..., 
        description="Unique identifier for this prediction request"
    )
    fraud_probability: float = Field(
        ..., 
        description="Probability of fraud (0.0 - 1.0)",
        ge=0.0,
        le=1.0
    )
    prediction: int = Field(
        ..., 
        description="Binary prediction (0: legitimate, 1: fraud)",
        ge=0,
        le=1
    )
    risk_level: RiskLevelEnum = Field(
        ...,
        description="Risk classification: LOW (<0.3), MEDIUM (0.3-0.7), HIGH (>0.7)"
    )
    top_features: List[FeatureContribution] = Field(
        ...,
        description="Top contributing features from SHAP analysis"
    )
    timestamp: datetime = Field(
        ...,
        description="Prediction timestamp (ISO 8601 format)"
    )
    model_version: Optional[str] = Field(
        None,
        description="Model identifier/version used for prediction"
    )


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class HealthResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "timestamp": "2026-02-27T10:30:00",
                "version": "1.0.0",
                "model_loaded": True,
                "h2o_connected": True,
                "uptime_seconds": 3600.5,
                "details": {
                    "h2o_version": "3.44.0.3",
                    "model_name": "XGBoost_1_AutoML_1_20260209_165338"
                }
            }
        }
    )
    
    status: HealthStatus = Field(
        ...,
        description="Overall service health status"
    )
    timestamp: datetime = Field(
        ...,
        description="Health check timestamp"
    )
    version: str = Field(
        ...,
        description="API version"
    )
    model_loaded: bool = Field(
        ...,
        description="Whether ML model is successfully loaded"
    )
    h2o_connected: bool = Field(
        ...,
        description="Whether H2O server is connected"
    )
    uptime_seconds: float = Field(
        ...,
        description="Service uptime in seconds"
    )
    details: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional health check details"
    )


class ErrorResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "Validation Error",
                "message": "Transaction amount must be positive",
                "request_id": "550e8400-e29b-41d4-a716-446655440000",
                "timestamp": "2026-02-27T10:30:00"
            }
        }
    )
    
    error: str = Field(..., description="Error type or category")
    message: str = Field(..., description="Human-readable error message")
    request_id: Optional[str] = Field(None, description="Request identifier for tracing")
    timestamp: datetime = Field(..., description="Error timestamp")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error context")
