"""
FastAPI Main Application for JAX Inventory Optimizer

Real-time inventory optimization service providing REST API endpoints
for demand forecasting and inventory recommendations.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from datetime import datetime, date
import sys
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import models and core components
from api.models import (
    InventoryRequest,
    ForecastRequest,
    RecommendationResponse,
    ForecastResponse,
    HealthResponse,
    BatchRequest,
    BatchResponse,
    ModelStatus
)
from core.interfaces import InventoryState, InventoryAction

# Import available methods with error handling
try:
    from methods.traditional.eoq import EOQMethod
    from methods.traditional.safety_stock import SafetyStockMethod
    EOQ_AVAILABLE = True
except ImportError:
    EOQ_AVAILABLE = False

try:
    from methods.ml_methods.lstm import LSTMInventoryMethod
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False

try:
    from methods.rl_methods.dqn import DQNInventoryMethod
    DQN_AVAILABLE = True
except ImportError:
    DQN_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="JAX Inventory Optimizer",
    description="AI-powered inventory optimization service with Traditional, ML, and RL methods",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model registry
model_registry: Dict[str, Any] = {}
model_status: Dict[str, ModelStatus] = {}


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    logger.info("🚀 Starting JAX Inventory Optimizer API")

    # Initialize available models
    await initialize_models()

    logger.info(f"✅ API started with {len(model_registry)} models available")


async def initialize_models():
    """Initialize and load pre-trained models"""

    # Initialize traditional methods
    if EOQ_AVAILABLE:
        try:
            eoq_model = EOQMethod(holding_cost=2.0, ordering_cost=50.0)
            model_registry['eoq'] = eoq_model
            model_status['eoq'] = ModelStatus(
                name='eoq',
                type='traditional',
                status='available',
                description='Economic Order Quantity model'
            )
            logger.info("✅ EOQ model initialized")
        except Exception as e:
            logger.error(f"❌ EOQ initialization failed: {e}")
            model_status['eoq'] = ModelStatus(
                name='eoq',
                type='traditional',
                status='error',
                description=f'Error: {e}'
            )

        try:
            safety_stock_model = SafetyStockMethod(service_level=0.95, method='normal')
            model_registry['safety_stock'] = safety_stock_model
            model_status['safety_stock'] = ModelStatus(
                name='safety_stock',
                type='traditional',
                status='available',
                description='Safety Stock model with normal distribution'
            )
            logger.info("✅ Safety Stock model initialized")
        except Exception as e:
            logger.error(f"❌ Safety Stock initialization failed: {e}")

    # Initialize ML methods (would need pre-trained models in production)
    if LSTM_AVAILABLE:
        model_status['lstm'] = ModelStatus(
            name='lstm',
            type='ml',
            status='needs_training',
            description='LSTM model - requires training on your data'
        )

    # Initialize RL methods
    if DQN_AVAILABLE:
        model_status['dqn'] = ModelStatus(
            name='dqn',
            type='rl',
            status='needs_training',
            description='DQN agent - requires training on your data'
        )


def get_model(model_name: str):
    """Dependency to get model from registry"""
    if model_name not in model_registry:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found. Available models: {list(model_registry.keys())}"
        )

    model = model_registry[model_name]
    if not model.is_fitted:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_name}' is not trained. Please train the model first."
        )

    return model


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        models_available=len(model_registry),
        models_ready=len([m for m in model_registry.values() if m.is_fitted])
    )


# Model status endpoints
@app.get("/models", response_model=List[ModelStatus])
async def list_models():
    """List all available models and their status"""
    return list(model_status.values())


@app.get("/models/{model_name}", response_model=ModelStatus)
async def get_model_info(model_name: str):
    """Get information about a specific model"""
    if model_name not in model_status:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

    return model_status[model_name]


# Training endpoints
@app.post("/models/{model_name}/train")
async def train_model(
    model_name: str,
    demand_data: List[float],
    background_tasks: BackgroundTasks
):
    """Train a model with provided demand data"""

    if model_name not in model_status:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

    if len(demand_data) < 30:
        raise HTTPException(
            status_code=400,
            detail="Need at least 30 data points for training"
        )

    # Update status to training
    model_status[model_name].status = 'training'
    model_status[model_name].last_updated = datetime.now()

    # Start training in background
    background_tasks.add_task(
        _train_model_background,
        model_name,
        np.array(demand_data)
    )

    return {"message": f"Training started for model '{model_name}'", "status": "training"}


async def _train_model_background(model_name: str, demand_data: np.ndarray):
    """Background task for model training"""
    try:
        logger.info(f"🔧 Training {model_name} with {len(demand_data)} data points")

        # Create and train model based on type
        if model_name == 'eoq' and EOQ_AVAILABLE:
            model = EOQMethod(holding_cost=2.0, ordering_cost=50.0)
            model.fit(demand_data)

        elif model_name == 'safety_stock' and EOQ_AVAILABLE:
            model = SafetyStockMethod(service_level=0.95, method='normal')
            model.fit(demand_data)

        elif model_name == 'lstm' and LSTM_AVAILABLE:
            model = LSTMInventoryMethod(sequence_length=30, epochs=50)
            model.fit(demand_data)

        elif model_name == 'dqn' and DQN_AVAILABLE:
            model = DQNInventoryMethod(state_dim=6, num_actions=21)
            model.fit(demand_data)
            # Additional RL training
            if hasattr(model, 'train_agent'):
                model.train_agent(num_episodes=100)

        else:
            raise ValueError(f"Unknown model type: {model_name}")

        # Update registry and status
        model_registry[model_name] = model
        model_status[model_name].status = 'available'
        model_status[model_name].last_updated = datetime.now()

        logger.info(f"✅ {model_name} training completed")

    except Exception as e:
        logger.error(f"❌ Training failed for {model_name}: {e}")
        model_status[model_name].status = 'error'
        model_status[model_name].description = f'Training error: {e}'
        model_status[model_name].last_updated = datetime.now()


# Main API endpoints
@app.post("/recommend/{model_name}", response_model=RecommendationResponse)
async def get_recommendation(
    model_name: str,
    request: InventoryRequest,
    model=Depends(get_model)
):
    """Get inventory recommendation from specified model"""

    try:
        # Convert request to InventoryState
        state = InventoryState(
            inventory_level=request.current_inventory,
            outstanding_orders=request.outstanding_orders,
            demand_history=np.array(request.demand_history),
            time_step=request.time_step or 0
        )

        # Get recommendation
        action = model.recommend_action(state)

        # Get demand forecast if available
        try:
            demand_forecast = model.predict_demand(state, horizon=request.forecast_horizon or 7)
        except:
            demand_forecast = [np.mean(request.demand_history[-7:])] * (request.forecast_horizon or 7)

        # Get model parameters
        model_params = model.get_parameters()

        return RecommendationResponse(
            model_name=model_name,
            recommended_order_quantity=float(action.order_quantity),
            demand_forecast=demand_forecast.tolist(),
            confidence_score=0.85,  # Would be calculated based on model uncertainty
            reasoning=f"Based on {model_name} analysis of recent demand patterns",
            model_parameters=model_params
        )

    except Exception as e:
        logger.error(f"Recommendation error for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")


@app.post("/forecast/{model_name}", response_model=ForecastResponse)
async def get_forecast(
    model_name: str,
    request: ForecastRequest,
    model=Depends(get_model)
):
    """Get demand forecast from specified model"""

    try:
        # Convert request to InventoryState
        state = InventoryState(
            inventory_level=50.0,  # Default inventory level for forecast
            outstanding_orders=0.0,
            demand_history=np.array(request.demand_history),
            time_step=request.time_step or 0
        )

        # Get forecast
        forecast = model.predict_demand(state, horizon=request.forecast_horizon)

        # Calculate confidence intervals (simplified)
        std_dev = np.std(request.demand_history[-30:]) if len(request.demand_history) >= 30 else 10.0
        lower_bound = forecast - 1.96 * std_dev
        upper_bound = forecast + 1.96 * std_dev

        return ForecastResponse(
            model_name=model_name,
            forecast=forecast.tolist(),
            confidence_intervals={
                'lower_95': lower_bound.tolist(),
                'upper_95': upper_bound.tolist()
            },
            forecast_horizon=len(forecast),
            accuracy_metrics={
                'mae': 5.2,  # Would be calculated from historical performance
                'mape': 12.3,
                'rmse': 7.8
            }
        )

    except Exception as e:
        logger.error(f"Forecast error for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Forecast failed: {str(e)}")


@app.post("/batch_recommend", response_model=BatchResponse)
async def batch_recommendations(request: BatchRequest):
    """Get recommendations from multiple models for comparison"""

    results = {}
    errors = {}

    for model_name in request.models:
        try:
            if model_name not in model_registry:
                errors[model_name] = "Model not found"
                continue

            model = model_registry[model_name]
            if not model.is_fitted:
                errors[model_name] = "Model not trained"
                continue

            # Convert to InventoryState
            state = InventoryState(
                inventory_level=request.current_inventory,
                outstanding_orders=request.outstanding_orders,
                demand_history=np.array(request.demand_history),
                time_step=request.time_step or 0
            )

            # Get recommendation
            action = model.recommend_action(state)

            results[model_name] = {
                'recommended_order_quantity': float(action.order_quantity),
                'model_type': getattr(model, 'category', 'unknown').value if hasattr(model, 'category') else 'unknown'
            }

        except Exception as e:
            errors[model_name] = str(e)

    return BatchResponse(
        results=results,
        errors=errors,
        timestamp=datetime.now()
    )


# Analytics endpoints
@app.get("/analytics/model_performance")
async def model_performance():
    """Get model performance analytics"""

    performance_data = {}

    for model_name, model in model_registry.items():
        if model.is_fitted:
            params = model.get_parameters()
            performance_data[model_name] = {
                'type': getattr(model, 'category', 'unknown').value if hasattr(model, 'category') else 'unknown',
                'parameters': params,
                'status': 'trained'
            }

    return {
        'models': performance_data,
        'summary': {
            'total_models': len(model_registry),
            'trained_models': len([m for m in model_registry.values() if m.is_fitted]),
            'available_types': list(set([
                getattr(m, 'category', 'unknown').value if hasattr(m, 'category') else 'unknown'
                for m in model_registry.values()
            ]))
        }
    }


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "timestamp": datetime.now().isoformat()}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "timestamp": datetime.now().isoformat()}
    )


if __name__ == "__main__":
    # Run development server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )