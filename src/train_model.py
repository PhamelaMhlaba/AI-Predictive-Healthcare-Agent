"""
Malaria Prediction Model Training Pipeline
Implementation of RandomForestRegressor training for malaria prediction.
"""

import pandas as pd
import numpy as np
import joblib
import logging
import sys
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
import json

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))
from model import MalariaPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('reports/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MalariaModelTrainer:
    """
    Training pipeline for Malaria Prediction using RandomForestRegressor.
    Handles data loading, model training, evaluation, and persistence.
    """
    
    DEFAULT_PARAMS = {
        'n_estimators': 200,
        'max_depth': 20,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': 42,
        'n_jobs': -1
    }
    
    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        self.model_params = model_params or self.DEFAULT_PARAMS
        self.model = None
        self.metrics = {}
        self.feature_importance = None

    
    def validate_data_paths(self) -> None:
        # Path to data folder relative to src
        data_dir = Path(__file__).parent.parent / "data/processed"
        
        # List of required files
        required_files = [
            data_dir / "X_train.csv",
            data_dir / "X_test.csv",
            data_dir / "y_train.csv",
            data_dir / "y_test.csv"
        ]
        
        # Check which files are missing
        missing_files = [str(f) for f in required_files if not f.exists()]

        if missing_files:
            error_msg = f"Missing data files: {missing_files}. Run data_preprocessor.py first."
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        logger.info("All required data files validated")
    
    def load_training_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        try:
            logger.info("Loading preprocessed training data...")
            
            X_train = pd.read_csv('../data/processed/X_train.csv')
            X_test = pd.read_csv('../data/processed/X_test.csv')
            y_train = pd.read_csv('../data/processed/y_train.csv').squeeze()
            y_test = pd.read_csv('../data/processed/y_test.csv').squeeze()
            
            self._validate_data_shapes(X_train, X_test, y_train, y_test)
            self._validate_data_consistency(X_train, X_test)
            
            logger.info(f"Data loaded - Train: {X_train.shape}, Test: {X_test.shape}, Features: {X_train.shape[1]}")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"Failed to load training data: {str(e)}")
            raise
    
    def _validate_data_shapes(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                            y_train: pd.Series, y_test: pd.Series) -> None:
        if len(X_train) != len(y_train):
            raise ValueError(f"X_train and y_train shape mismatch: {X_train.shape} vs {y_train.shape}")
        if len(X_test) != len(y_test):
            raise ValueError(f"X_test and y_test shape mismatch: {X_test.shape} vs {y_test.shape}")
        if X_train.shape[1] != X_test.shape[1]:
            raise ValueError(f"Feature dimension mismatch: train {X_train.shape[1]} vs test {X_test.shape[1]}")
    
    def _validate_data_consistency(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> None:
        if list(X_train.columns) != list(X_test.columns):
            raise ValueError("Training and test features do not match")
        
        if X_train.isnull().any().any():
            raise ValueError("Training data contains NaN values")
        if X_test.isnull().any().any():
            raise ValueError("Test data contains NaN values")
    
    def initialize_model(self) -> MalariaPredictor:
        try:
            logger.info(f"Initializing RandomForest model with parameters: {self.model_params}")
            model = MalariaPredictor(**self.model_params)
            return model
            
        except Exception as e:
            logger.error(f"Model initialization failed: {str(e)}")
            raise
    
    def perform_cross_validation(self, model: MalariaPredictor, X_train: pd.DataFrame, 
                               y_train: pd.Series, cv_folds: int = 5) -> Dict[str, float]:
        try:
            logger.info(f"Performing {cv_folds}-fold cross-validation...")
            
            cv_scores = cross_val_score(
                model.model, X_train, y_train, 
                cv=cv_folds, scoring='r2', n_jobs=-1
            )
            
            cv_metrics = {
                'cv_r2_mean': cv_scores.mean(),
                'cv_r2_std': cv_scores.std(),
                'cv_r2_scores': cv_scores.tolist()
            }
            
            logger.info(f"Cross-validation R²: {cv_metrics['cv_r2_mean']:.4f} (±{cv_metrics['cv_r2_std']:.4f})")
            
            return cv_metrics
            
        except Exception as e:
            logger.warning(f"Cross-validation failed: {str(e)}")
            return {}
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> MalariaPredictor:
        try:
            logger.info("Starting model training...")
            
            self.model = self.initialize_model()
            self.model.fit(X_train, y_train)
            
            cv_metrics = self.perform_cross_validation(self.model, X_train, y_train)
            self.metrics.update(cv_metrics)
            
            logger.info("Model training completed successfully")
            return self.model
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            raise
    
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        try:
            logger.info("Evaluating model on test data...")
            
            predictions = self.model.predict(X_test)
            
            self.metrics.update({
                'mse': mean_squared_error(y_test, predictions),
                'mae': mean_absolute_error(y_test, predictions),
                'r2': r2_score(y_test, predictions),
                'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
                'test_samples': len(X_test),
                'training_samples': self.model.training_shape[0] if hasattr(self.model, 'training_shape') else 0
            })
            
            logger.info(f"Test R² Score: {self.metrics['r2']:.4f}")
            logger.info(f"Test RMSE: {self.metrics['rmse']:.4f}")
            logger.info(f"Test MAE: {self.metrics['mae']:.4f}")
            
            return self.metrics
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {str(e)}")
            raise
    
    def analyze_feature_importance(self) -> Optional[pd.Series]:
        if self.model is None:
            return None
        
        try:
            self.feature_importance = self.model.get_feature_importance()
            
            if self.feature_importance is not None:
                logger.info("Top 10 Most Important Features:")
                for feature, importance in self.feature_importance.head(10).items():
                    logger.info(f"  {feature}: {importance:.4f}")
            
            return self.feature_importance
            
        except Exception as e:
            logger.warning(f"Feature importance analysis failed: {str(e)}")
            return None
    
    def save_model_artifacts(self, model_path: str = 'models/random_forest_model.pkl') -> None:
        if self.model is None:
            raise ValueError("No trained model to save")
        
        try:
            Path('models').mkdir(exist_ok=True)
            Path('reports').mkdir(exist_ok=True)
            
            self.model.save_model(model_path)
            
            with open('reports/training.log', 'w') as f:
                json.dump(self.metrics, f, indent=2)
            
            if self.feature_importance is not None:
                self.feature_importance.to_csv('reports/feature_importance.csv', header=True)

            self._generate_performance_report()
            
            logger.info(f"Model artifacts saved successfully")
            logger.info(f"Model: {model_path}")
            logger.info("Metrics: ../reports/training_metrics.json")
            logger.info("Feature importance: ../reports/feature_importance.csv")
            
        except Exception as e:
            logger.error(f"Failed to save model artifacts: {str(e)}")
            raise
    
    def _generate_performance_report(self) -> None:
        try:
            report_content = f"""# Model Performance Report

## Model Details
- **Algorithm**: RandomForestRegressor
- **Training Samples**: {self.metrics.get('training_samples', 'N/A')}
- **Test Samples**: {self.metrics.get('test_samples', 'N/A')}
- **Number of Features**: {self.model.training_shape[1] if self.model and hasattr(self.model, 'training_shape') else 'N/A'}

## Performance Metrics
- **R² Score**: {self.metrics.get('r2', 0):.4f}
- **RMSE**: {self.metrics.get('rmse', 0):.4f}
- **MAE**: {self.metrics.get('mae', 0):.4f}
- **MSE**: {self.metrics.get('mse', 0):.4f}

## Cross-Validation
- **CV R² Mean**: {self.metrics.get('cv_r2_mean', 0):.4f}
- **CV R² Std**: {self.metrics.get('cv_r2_std', 0):.4f}

## Acceptance Criteria
- **R² > 0.85**: {'PASS' if self.metrics.get('r2', 0) > 0.85 else 'FAIL'}

## Model Parameters.
{json.dumps(self.model_params, indent=2)}
"""
            
            with open('../reports/model_performance.md', 'w') as f:
                f.write(report_content)
                
        except Exception as e:
            logger.warning(f"Failed to generate performance report: {str(e)}")
    
    def run_training_pipeline(self) -> Dict[str, Any]:
        try:
            logger.info("Starting Malaria Prediction training pipeline...")
            
            self.validate_data_paths()
            X_train, X_test, y_train, y_test = self.load_training_data()
            trained_model = self.train_model(X_train, y_train)
            metrics = self.evaluate_model(X_test, y_test)
            feature_importance = self.analyze_feature_importance()
            self.save_model_artifacts()
            
            r2_score = metrics.get('r2', 0)
            if r2_score > 0.85:
                logger.info("Model meets acceptance criteria (R² > 0.85)")
            else:
                logger.warning(f"Model below acceptance criteria (R² = {r2_score:.4f})")
            
            logger.info("Training pipeline completed successfully")
            
            return {
                'model': trained_model,
                'metrics': metrics,
                'feature_importance': feature_importance,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }


def main():
    try:
        trainer = MalariaModelTrainer()
        results = trainer.run_training_pipeline()
        
        if results['success']:
            print("\n" + "="*60)
            print("MALARIA PREDICTION MODEL TRAINING - COMPLETED SUCCESSFULLY")
            print("="*60)
            print(f"Final R² Score: {results['metrics']['r2']:.4f}")
            print(f"RMSE: {results['metrics']['rmse']:.4f}")
            print(f"Model saved: ../models/random_forest_model.pkl")
            print("="*60)
            sys.exit(0)
        else:
            print(f"\nTraining failed: {results.get('error', 'Unknown error')}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Training execution failed: {str(e)}")
        print(f"\nCritical error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()