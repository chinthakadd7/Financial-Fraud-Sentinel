"""
Fraud Prediction Model Handler.

Manages H2O AutoML model lifecycle, predictions, and explainability features.
Implements lazy initialization and health checking for production deployment.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
import socket

# Add project root to Python path (needed when running from different contexts)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import h2o
import pandas as pd
import warnings

from src.data_pipeline.preprocess import preprocess_input

# Suppress H2O warnings about missing columns
warnings.filterwarnings('ignore', category=UserWarning, module='h2o')
warnings.filterwarnings('ignore', category=Warning, module='h2o')

# Try to import config and logger, fallback if not available
try:
    from src.config import settings
    MODEL_PATH = settings.MODEL_PATH
    H2O_MEMORY = settings.h2o_memory_str
    H2O_PORT = settings.H2O_PORT
    H2O_NTHREADS = settings.H2O_NTHREADS
except ImportError:
    # Fallback defaults
    MODEL_PATH = os.path.join(
        os.path.dirname(__file__),
        "../../models_artifacts/XGBoost_1_AutoML_1_20260209_165338.zip"
    )
    H2O_MEMORY = "2G"
    H2O_PORT = 54321
    H2O_NTHREADS = -1

try:
    from src.utils.logger import model_logger as logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)


class FraudPredictor:
    """
    H2O-based fraud prediction service with health monitoring.
    
    Implements lazy initialization pattern to defer H2O server startup
    until first prediction, reducing startup overhead.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize predictor (without starting H2O).
        
        Args:
            model_path: Path to H2O MOJO model file. Uses config default if None.
        """
        self.model_path = Path(model_path or MODEL_PATH)
        self.model = None
        self.h2o_initialized = False
        self.initialization_time: Optional[float] = None
        self.startup_timestamp: Optional[float] = None
        
        logger.info(f"FraudPredictor created with model path: {self.model_path}")
    
    def _is_port_in_use(self, port: int) -> bool:
        """Check if a port is already in use."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0
    
    def _ensure_h2o_initialized(self):
        """
        Initialize H2O server and load model (lazy initialization).
        
        Raises:
            RuntimeError: If model file doesn't exist or initialization fails
        """
        if self.h2o_initialized and self.model is not None:
            return
        
        start_time = time.time()
        
        try:
            # Validate model path
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"Model file not found: {self.model_path}. "
                    f"Please ensure the model is present in models_artifacts/"
                )
            
            # Check if H2O is already running
            if self._is_port_in_use(H2O_PORT):
                logger.warning(f"H2O port {H2O_PORT} already in use, attempting to connect to existing instance")
                try:
                    h2o.init(
                        port=H2O_PORT,
                        start_h2o=False,
                        strict_version_check=False
                    )
                except Exception as e:
                    logger.error(f"Failed to connect to existing H2O instance: {e}")
                    raise RuntimeError(f"H2O port conflict on {H2O_PORT}. Please stop existing H2O instance.")
            else:
                # Start new H2O instance
                logger.info(f"Initializing H2O server with {H2O_MEMORY} memory on port {H2O_PORT}")
                h2o.init(
                    max_mem_size=H2O_MEMORY,
                    port=H2O_PORT,
                    nthreads=H2O_NTHREADS,
                    start_h2o=True,
                    strict_version_check=False
                )
            
            self.h2o_initialized = True
            logger.info("H2O server initialized successfully")
            
            # Load MOJO model
            logger.info(f"Loading model from {self.model_path}")
            self.model = h2o.import_mojo(str(self.model_path))
            
            self.initialization_time = time.time() - start_time
            self.startup_timestamp = time.time()
            
            logger.info(
                f"Model loaded successfully in {self.initialization_time:.2f}s",
                extra={"duration_ms": round(self.initialization_time * 1000, 2)}
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize predictor: {e}", exc_info=True)
            self.h2o_initialized = False
            self.model = None
            raise RuntimeError(f"Predictor initialization failed: {e}")
    
    def health_check(self) -> Dict[str, any]:
        """
        Check health status of predictor and H2O server.
        
        Returns:
            Dictionary with health status information
        """
        health_info = {
            "model_loaded": self.model is not None,
            "h2o_initialized": self.h2o_initialized,
            "model_path": str(self.model_path),
            "model_exists": self.model_path.exists(),
        }
        
        # Check H2O connection
        if self.h2o_initialized:
            try:
                # Simple check: get cluster info
                cluster_info = h2o.cluster().show_status(False)
                health_info["h2o_connected"] = True
                health_info["h2o_version"] = h2o.__version__
            except Exception as e:
                health_info["h2o_connected"] = False
                health_info["h2o_error"] = str(e)
                logger.error(f"H2O health check failed: {e}")
        else:
            health_info["h2o_connected"] = False
        
        # Add uptime if model is loaded
        if self.startup_timestamp:
            health_info["uptime_seconds"] = round(time.time() - self.startup_timestamp, 2)
        
        return health_info
    
    def predict_transaction(self, transaction: dict) -> Dict[str, any]:
        """
        Predict fraud probability for a single transaction.
        
        Args:
            transaction: Dictionary containing transaction features
        
        Returns:
            Dictionary with prediction results:
                - fraud_probability: float (0.0-1.0)
                - prediction: int (0 or 1)
                - top_features: dict of {feature: contribution}
        
        Raises:
            RuntimeError: If prediction fails
        """
        try:
            # Ensure H2O and model are initialized
            self._ensure_h2o_initialized()
            
            # Preprocess input
            logger.debug(f"Preprocessing transaction")
            processed_transaction = preprocess_input(transaction)
            
            # Convert to pandas DataFrame first to control data types
            df = pd.DataFrame([processed_transaction])
            
            # Ensure numeric columns are properly typed (not categorical)
            numeric_cols = [
                'TransactionAmt', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 
                'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14',
                'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 
                'D11', 'D12', 'D13', 'D14', 'D15',
                'card1', 'card2', 'card3', 'card5', 'addr1', 'addr2',
                'is_large_amount', 'transaction_count'
            ]
            
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convert to H2OFrame and predict
            logger.debug("Converting to H2OFrame and generating prediction")
            hf = h2o.H2OFrame(df)
            preds = self.model.predict(hf)
            result = preds.as_data_frame()
            
            # SHAP Contributions for explainability
            logger.debug("Calculating SHAP contributions")
            contrib = self.model.predict_contributions(hf)
            contrib_df = contrib.as_data_frame()
            
            # Remove bias term
            contrib_df = contrib_df.drop(columns=["BiasTerm"], errors="ignore")
            
            # Sort by absolute contribution
            contrib_series = contrib_df.iloc[0].abs().sort_values(ascending=False)
            
            # Get top 3 features
            top_features = contrib_series.head(3).to_dict()
            
            fraud_prob = float(result["p1"][0])
            prediction = int(result["predict"][0])
            
            logger.info(
                f"Prediction completed: fraud_prob={fraud_prob:.3f}, prediction={prediction}",
                extra={"fraud_probability": fraud_prob, "prediction": prediction}
            )
            
            return {
                "fraud_probability": fraud_prob,
                "prediction": prediction,
                "top_features": top_features
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}", exc_info=True)
            raise RuntimeError(f"Prediction failed: {e}")
    
    def warmup(self):
        """
        Warm up the model by initializing H2O and loading the model.
        
        Useful for reducing first-request latency in production.
        """
        logger.info("Warming up predictor...")
        try:
            self._ensure_h2o_initialized()
            logger.info("Predictor warmup completed successfully")
        except Exception as e:
            logger.error(f"Predictor warmup failed: {e}")
            raise
    
    def shutdown(self):
        """
        Gracefully shutdown H2O server.
        
        Should be called during application shutdown to cleanup resources.
        """
        if self.h2o_initialized:
            try:
                logger.info("Shutting down H2O server")
                h2o.cluster().shutdown()
                self.h2o_initialized = False
                self.model = None
                logger.info("H2O server shutdown complete")
            except Exception as e:
                logger.error(f"Error during H2O shutdown: {e}")


# Global predictor instance (lazy initialization)
_predictor: Optional[FraudPredictor] = None


def get_predictor() -> FraudPredictor:
    """
    Get or create global predictor instance.
    
    Implements singleton pattern for efficient resource usage.
    
    Returns:
        FraudPredictor instance
    """
    global _predictor
    if _predictor is None:
        _predictor = FraudPredictor()
    return _predictor


def predict_transaction(transaction: dict) -> Dict[str, any]:
    """
    Convenience function for backward compatibility.
    
    Takes a single transaction dict and returns fraud probability.
    
    Args:
        transaction: Dictionary containing transaction features
    
    Returns:
        Dictionary with prediction results
    """
    predictor = get_predictor()
    return predictor.predict_transaction(transaction)