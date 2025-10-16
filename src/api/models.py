"""
Pydantic Models for FastAPI

Data models for request/response validation and documentation.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any, Union
from datetime import datetime
from enum import Enum


class ModelType(str, Enum):
    """Available model types"""
    traditional = "traditional"
    ml = "ml"
    rl = "rl"


class ModelStatusEnum(str, Enum):
    """Model status options"""
    available = "available"
    training = "training"
    needs_training = "needs_training"
    error = "error"


# Request Models
class InventoryRequest(BaseModel):
    """Request for inventory recommendation"""
    current_inventory: float = Field(..., ge=0, description="Current inventory level")
    outstanding_orders: float = Field(0, ge=0, description="Orders in transit")
    demand_history: List[float] = Field(..., min_items=7, description="Historical demand data")
    time_step: Optional[int] = Field(None, description="Current time step")
    forecast_horizon: Optional[int] = Field(7, ge=1, le=30, description="Forecast horizon in periods")

    @validator('demand_history')
    def validate_demand_history(cls, v):
        if len(v) < 7:
            raise ValueError('Need at least 7 historical demand points')
        if any(x < 0 for x in v):
            raise ValueError('Demand values must be non-negative')
        return v

    class Config:
        schema_extra = {
            "example": {
                "current_inventory": 75.0,
                "outstanding_orders": 25.0,
                "demand_history": [45, 52, 48, 61, 55, 49, 58, 53, 47, 64],
                "time_step": 365,
                "forecast_horizon": 7
            }
        }


class ForecastRequest(BaseModel):
    """Request for demand forecasting"""
    demand_history: List[float] = Field(..., min_items=7, description="Historical demand data")
    forecast_horizon: int = Field(7, ge=1, le=30, description="Number of periods to forecast")
    time_step: Optional[int] = Field(None, description="Current time step")
    include_confidence: bool = Field(True, description="Include confidence intervals")

    class Config:
        schema_extra = {
            "example": {
                "demand_history": [45, 52, 48, 61, 55, 49, 58, 53, 47, 64, 51, 56],
                "forecast_horizon": 14,
                "time_step": 365,
                "include_confidence": True
            }
        }


class BatchRequest(BaseModel):
    """Request for batch recommendations from multiple models"""
    models: List[str] = Field(..., min_items=1, description="List of model names")
    current_inventory: float = Field(..., ge=0, description="Current inventory level")
    outstanding_orders: float = Field(0, ge=0, description="Orders in transit")
    demand_history: List[float] = Field(..., min_items=7, description="Historical demand data")
    time_step: Optional[int] = Field(None, description="Current time step")

    class Config:
        schema_extra = {
            "example": {
                "models": ["eoq", "safety_stock", "lstm"],
                "current_inventory": 75.0,
                "outstanding_orders": 25.0,
                "demand_history": [45, 52, 48, 61, 55, 49, 58, 53, 47, 64],
                "time_step": 365
            }
        }


# Response Models
class RecommendationResponse(BaseModel):
    """Response for inventory recommendation"""
    model_name: str = Field(..., description="Name of the model used")
    recommended_order_quantity: float = Field(..., description="Recommended order quantity")
    demand_forecast: List[float] = Field(..., description="Demand forecast for the horizon")
    confidence_score: float = Field(..., ge=0, le=1, description="Confidence in recommendation")
    reasoning: str = Field(..., description="Explanation of the recommendation")
    model_parameters: Dict[str, Any] = Field(default_factory=dict, description="Model parameters used")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")

    class Config:
        schema_extra = {
            "example": {
                "model_name": "lstm",
                "recommended_order_quantity": 95.5,
                "demand_forecast": [52.3, 48.7, 55.1, 49.8, 53.6, 51.2, 47.9],
                "confidence_score": 0.87,
                "reasoning": "Based on LSTM analysis of recent demand patterns showing weekly seasonality",
                "model_parameters": {
                    "sequence_length": 30,
                    "hidden_size": 64,
                    "forecast_horizon": 7
                },
                "timestamp": "2024-01-15T10:30:45"
            }
        }


class ForecastResponse(BaseModel):
    """Response for demand forecasting"""
    model_name: str = Field(..., description="Name of the model used")
    forecast: List[float] = Field(..., description="Demand forecast values")
    confidence_intervals: Optional[Dict[str, List[float]]] = Field(
        None, description="Confidence intervals (e.g., lower_95, upper_95)"
    )
    forecast_horizon: int = Field(..., description="Number of periods forecasted")
    accuracy_metrics: Optional[Dict[str, float]] = Field(
        None, description="Historical accuracy metrics"
    )
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")

    class Config:
        schema_extra = {
            "example": {
                "model_name": "transformer",
                "forecast": [52.3, 48.7, 55.1, 49.8, 53.6, 51.2, 47.9],
                "confidence_intervals": {
                    "lower_95": [42.1, 38.5, 44.9, 39.6, 43.4, 41.0, 37.7],
                    "upper_95": [62.5, 58.9, 65.3, 60.0, 63.8, 61.4, 58.1]
                },
                "forecast_horizon": 7,
                "accuracy_metrics": {
                    "mae": 4.2,
                    "mape": 8.5,
                    "rmse": 6.1
                },
                "timestamp": "2024-01-15T10:30:45"
            }
        }


class BatchResponse(BaseModel):
    """Response for batch recommendations"""
    results: Dict[str, Dict[str, Any]] = Field(..., description="Results for each model")
    errors: Dict[str, str] = Field(default_factory=dict, description="Errors for failed models")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")

    class Config:
        schema_extra = {
            "example": {
                "results": {
                    "eoq": {
                        "recommended_order_quantity": 87.3,
                        "model_type": "traditional"
                    },
                    "lstm": {
                        "recommended_order_quantity": 95.5,
                        "model_type": "ml"
                    }
                },
                "errors": {
                    "dqn": "Model not trained"
                },
                "timestamp": "2024-01-15T10:30:45"
            }
        }


class ModelStatus(BaseModel):
    """Status information for a model"""
    name: str = Field(..., description="Model name")
    type: ModelType = Field(..., description="Model type")
    status: ModelStatusEnum = Field(..., description="Current status")
    description: str = Field(..., description="Model description")
    last_updated: Optional[datetime] = Field(None, description="Last update timestamp")
    version: Optional[str] = Field(None, description="Model version")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Model parameters")

    class Config:
        schema_extra = {
            "example": {
                "name": "lstm",
                "type": "ml",
                "status": "available",
                "description": "LSTM neural network for demand forecasting",
                "last_updated": "2024-01-15T09:15:30",
                "version": "1.0.0",
                "parameters": {
                    "sequence_length": 30,
                    "hidden_size": 64,
                    "num_layers": 2
                }
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.now, description="Check timestamp")
    models_available: int = Field(..., description="Number of available models")
    models_ready: int = Field(..., description="Number of trained/ready models")
    version: str = Field("1.0.0", description="API version")

    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-15T10:30:45",
                "models_available": 5,
                "models_ready": 3,
                "version": "1.0.0"
            }
        }


class ErrorResponse(BaseModel):
    """Error response model"""
    detail: str = Field(..., description="Error description")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    error_code: Optional[str] = Field(None, description="Error code")

    class Config:
        schema_extra = {
            "example": {
                "detail": "Model 'unknown_model' not found",
                "timestamp": "2024-01-15T10:30:45",
                "error_code": "MODEL_NOT_FOUND"
            }
        }


# Training Models
class TrainingRequest(BaseModel):
    """Request for model training"""
    demand_data: List[float] = Field(..., min_items=30, description="Training demand data")
    validation_split: float = Field(0.2, ge=0.1, le=0.3, description="Validation data fraction")
    hyperparameters: Optional[Dict[str, Any]] = Field(None, description="Custom hyperparameters")

    @validator('demand_data')
    def validate_training_data(cls, v):
        if len(v) < 30:
            raise ValueError('Need at least 30 data points for training')
        if any(x < 0 for x in v):
            raise ValueError('Demand values must be non-negative')
        return v

    class Config:
        schema_extra = {
            "example": {
                "demand_data": [45, 52, 48, 61, 55] * 20,  # 100 data points
                "validation_split": 0.2,
                "hyperparameters": {
                    "learning_rate": 0.001,
                    "epochs": 100,
                    "batch_size": 32
                }
            }
        }


class TrainingResponse(BaseModel):
    """Response for training request"""
    message: str = Field(..., description="Training status message")
    training_id: str = Field(..., description="Training job ID")
    status: str = Field(..., description="Training status")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")

    class Config:
        schema_extra = {
            "example": {
                "message": "Training started successfully",
                "training_id": "train_lstm_20240115_103045",
                "status": "started",
                "estimated_completion": "2024-01-15T11:00:00"
            }
        }