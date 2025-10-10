"""
Malaria Prediction Model Architecture
Defines the core model class for malaria prediction using RandomForestRegressor.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import NotFittedError
import logging
from typing import Union, Optional, Dict, Any
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelNotTrainedError(Exception):
    """Custom exception for untrained model usage."""
    pass


class MalariaPredictor:
    """
    Malaria prediction model using RandomForestRegressor.
    
    This class provides a clean interface for model initialization,
    training, prediction, and persistence with proper error handling
    and validation.
    """
    
    def __init__(self, n_estimators: int = 100, random_state: int = 42, 
                 **kwargs: Any) -> None:
        """
        Initialize the Malaria Prediction Model.
        
        Args:
            n_estimators: Number of trees in the random forest
            random_state: Random seed for reproducibility
            **kwargs: Additional RandomForestRegressor parameters
        
        Example:
            >>> model = MalariaPredictor(n_estimators=200, max_depth=10)
        """
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model_kwargs = kwargs
        
        # Initialize model with parameters
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            **kwargs
        )
        
        self.is_trained = False
        self.feature_names = None
        self.training_shape = None
        
        logger.info(f"MalariaPredictor initialized with {n_estimators} estimators")
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], 
            y: Union[pd.Series, np.ndarray]) -> 'MalariaPredictor':
        """
        Train the model on the provided data.
        
        Args:
            X: Training features
            y: Training targets
            
        Returns:
            self: Trained model instance
            
        Raises:
            ValueError: If input data is invalid
        """
        try:
            # Validate input data
            if len(X) == 0 or len(y) == 0:
                raise ValueError("Training data cannot be empty")
            
            if len(X) != len(y):
                raise ValueError("X and y must have the same number of samples")
            
            # Store feature names if available
            if hasattr(X, 'columns'):
                self.feature_names = list(X.columns)
            
            # Train the model
            logger.info(f"Training model on {len(X)} samples...")
            self.model.fit(X, y)
            self.is_trained = True
            self.training_shape = X.shape
            
            logger.info("Model training completed successfully")
            return self
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Generate predictions using the trained model.
        
        Args:
            X: Input features for prediction
            
        Returns:
            np.ndarray: Model predictions
            
        Raises:
            ModelNotTrainedError: If model hasn't been trained
            ValueError: If input data is invalid
        """
        if not self.is_trained:
            raise ModelNotTrainedError(
                "Model must be trained before making predictions. Call fit() first."
            )
        
        if len(X) == 0:
            raise ValueError("Input data cannot be empty")
        
        # Validate feature dimensions if training shape is known
        if self.training_shape and X.shape[1] != self.training_shape[1]:
            raise ValueError(
                f"Input has {X.shape[1]} features, "
                f"but model expects {self.training_shape[1]} features"
            )
        
        try:
            predictions = self.model.predict(X)
            logger.debug(f"Generated predictions for {len(X)} samples")
            return predictions
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise
    
    def get_feature_importance(self) -> Optional[pd.Series]:
        """
        Get feature importance scores from the trained model.
        
        Returns:
            pd.Series: Feature importance scores, or None if model not trained
        """
        if not self.is_trained:
            logger.warning("Cannot get feature importance: model not trained")
            return None
        
        try:
            importance = self.model.feature_importances_
            if self.feature_names and len(self.feature_names) == len(importance):
                return pd.Series(importance, index=self.feature_names).sort_values(ascending=False)
            return pd.Series(importance).sort_values(ascending=False)
            
        except Exception as e:
            logger.error(f"Failed to get feature importance: {str(e)}")
            return None
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            'n_estimators': self.n_estimators,
            'random_state': self.random_state,
            **self.model_kwargs
        }
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path where model will be saved
            
        Raises:
            ModelNotTrainedError: If model hasn't been trained
        """
        if not self.is_trained:
            raise ModelNotTrainedError("Cannot save untrained model")
        
        try:
            joblib.dump(self, filepath)
            logger.info(f"Model saved successfully to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise
    
    @classmethod
    def load_model(cls, filepath: str) -> 'MalariaPredictor':
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to saved model file
            
        Returns:
            MalariaPredictor: Loaded model instance
        """
        try:
            model = joblib.load(filepath)
            if not isinstance(model, MalariaPredictor):
                raise ValueError("Loaded object is not a MalariaPredictor instance")
            
            logger.info(f"Model loaded successfully from {filepath}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def __repr__(self) -> str:
        """String representation of the model."""
        status = "trained" if self.is_trained else "untrained"
        return (f"MalariaPredictor(n_estimators={self.n_estimators}, "
                f"random_state={self.random_state}, status='{status}')")


# Example usage and testing
if __name__ == "__main__":
    # Quick smoke test
    print("Testing MalariaPredictor...")
    
    # Create model instance
    model = MalariaPredictor(n_estimators=50)
    print(f"Created: {model}")
    
    # Test parameters
    print(f"Parameters: {model.get_params()}")
    
    # Test untrained prediction (should raise error)
    try:
        model.predict([[1, 2, 3]])
    except ModelNotTrainedError as e:
        print(f"✓ Correctly caught untrained prediction: {e}")
    
    print("Model class implementation successful!")