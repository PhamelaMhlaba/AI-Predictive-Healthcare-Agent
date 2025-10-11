"""
Malaria Prediction Streamlit Application
User interface for malaria prediction using trained RandomForest model.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from model import MalariaPredictor
except ImportError as e:
    st.error(f"Import error: {e}. Make sure model.py is in the src directory.")
    st.stop()


class MalariaPredictionApp:
    """
    Streamlit application for malaria prediction using trained ML model.
    """
    
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.model_metrics = {}
        self.feature_importance = None
        
    def load_model(self) -> bool:
        """
        Load the trained model and associated artifacts.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            model_path = project_root / "models" / "random_forest_model.pkl"
            
            if not model_path.exists():
                st.error(f"Model file not found at {model_path}. Please run train_model.py first.")
                return False
            
            self.model = MalariaPredictor.load_model(str(model_path))
            
            # Load model metrics if available
            metrics_path = project_root / "reports" / "training_metrics.json"
            if metrics_path.exists():
                import json
                with open(metrics_path, 'r') as f:
                    self.model_metrics = json.load(f)
            
            # Load feature importance if available
            feature_path = project_root / "reports" / "feature_importance.csv"
            if feature_path.exists():
                self.feature_importance = pd.read_csv(feature_path, index_col=0)
                self.feature_names = self.feature_importance.index.tolist()
            
            st.success(" Model loaded successfully!")
            return True
            
        except Exception as e:
            st.error(f"Failed to load model: {str(e)}")
            return False
    
    def render_sidebar(self) -> Dict[str, Any]:
        """
        Render the sidebar with model information and inputs.
        
        Returns:
            Dict containing user inputs and configuration
        """
        st.sidebar.title("🔬 Malaria Prediction Model")
        
        # Model information
        st.sidebar.markdown("### Model Information")
        if self.model_metrics:
            st.sidebar.metric("R² Score", f"{self.model_metrics.get('r2', 0):.3f}")
            st.sidebar.metric("RMSE", f"{self.model_metrics.get('rmse', 0):.3f}")
        
        st.sidebar.markdown("---")
        
        # Input method selection
        input_method = st.sidebar.radio(
            "Input Method",
            ["Single Prediction", "Batch Prediction (CSV)"],
            help="Choose between single patient prediction or batch prediction from CSV file"
        )
        
        return {"input_method": input_method}
    
    def render_single_prediction(self) -> Optional[pd.DataFrame]:
        """
        Render single prediction input form.
        
        Returns:
            DataFrame with input data for prediction, or None if not ready
        """
        st.header("Single Patient Prediction")
        
        # Create input form
        with st.form("prediction_form"):
            st.markdown("### Patient Information")
            
            # Example feature inputs - these should match your actual model features
            col1, col2 = st.columns(2)
            
            with col1:
                # Replace these with your actual feature names from the model
                age = st.number_input("Age", min_value=0, max_value=120, value=30)
                temperature = st.number_input("Temperature (°C)", min_value=35.0, max_value=42.0, value=37.0)
                heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=72)
            
            with col2:
                # Replace these with your actual feature names from the model
                blood_pressure = st.number_input("Blood Pressure (systolic)", min_value=80, max_value=200, value=120)
                respiratory_rate = st.number_input("Respiratory Rate", min_value=8, max_value=40, value=16)
                hemoglobin = st.number_input("Hemoglobin (g/dL)", min_value=5.0, max_value=20.0, value=14.0)
            
            # Add more features as needed based on your model
            
            submitted = st.form_submit_button("Predict Malaria Risk")
            
            if submitted:
                # Create input dataframe matching training features
                input_data = pd.DataFrame({
                    'age': [age],
                    'temperature': [temperature],
                    'heart_rate': [heart_rate],
                    'blood_pressure': [blood_pressure],
                    'respiratory_rate': [respiratory_rate],
                    'hemoglobin': [hemoglobin]
                    # Add more features to match your training data
                })
                
                return input_data
        
        return None
    
    def render_batch_prediction(self) -> Optional[pd.DataFrame]:
        """
        Render batch prediction file upload.
        
        Returns:
            DataFrame with uploaded data, or None if no file uploaded
        """
        st.header("Batch Prediction")
        
        st.markdown("""
        Upload a CSV file with patient data. The file should include the same features 
        used during model training.
        """)
        
        uploaded_file = st.file_uploader(
            "Choose CSV file",
            type="csv",
            help="CSV file with patient data for batch prediction"
        )
        
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.success(f"✅ Successfully loaded {len(data)} records")
                
                # Display preview
                st.markdown("### Data Preview")
                st.dataframe(data.head())
                
                st.markdown("### Data Summary")
                st.write(f"**Records:** {len(data)}")
                st.write(f"**Features:** {len(data.columns)}")
                
                return data
                
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                return None
        
        return None
    
    def make_prediction(self, input_data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the loaded model.
        
        Args:
            input_data: Input features for prediction
            
        Returns:
            Array of predictions
        """
        try:
            predictions = self.model.predict(input_data)
            return predictions
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            raise
    
    def display_single_prediction_result(self, prediction: float, input_data: pd.DataFrame):
        """
        Display single prediction results.
        
        Args:
            prediction: Single prediction value
            input_data: Input features used for prediction
        """
        st.header("Prediction Results")
        
        # Create risk assessment
        risk_level = "Low" if prediction < 0.3 else "Medium" if prediction < 0.7 else "High"
        risk_color = "green" if risk_level == "Low" else "orange" if risk_level == "Medium" else "red"
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Malaria Risk Score", f"{prediction:.3f}")
        
        with col2:
            st.metric("Risk Level", risk_level)
        
        with col3:
            st.metric("Confidence", "High" if abs(prediction - 0.5) > 0.3 else "Medium")
        
        # Risk gauge visualization
        self.display_risk_gauge(prediction)
        
        # Feature importance for this prediction (if available)
        if self.feature_importance is not None:
            self.display_feature_importance()
    
    def display_batch_prediction_results(self, predictions: np.ndarray, input_data: pd.DataFrame):
        """
        Display batch prediction results.
        
        Args:
            predictions: Array of predictions
            input_data: Input features used for predictions
        """
        st.header("Batch Prediction Results")
        
        # Add predictions to data
        results_df = input_data.copy()
        results_df['Malaria_Risk_Score'] = predictions
        results_df['Risk_Level'] = results_df['Malaria_Risk_Score'].apply(
            lambda x: 'Low' if x < 0.3 else 'Medium' if x < 0.7 else 'High'
        )
        
        # Display results table
        st.markdown("### Prediction Results")
        st.dataframe(results_df)
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            low_risk = (results_df['Risk_Level'] == 'Low').sum()
            st.metric("Low Risk Patients", low_risk)
        
        with col2:
            medium_risk = (results_df['Risk_Level'] == 'Medium').sum()
            st.metric("Medium Risk Patients", medium_risk)
        
        with col3:
            high_risk = (results_df['Risk_Level'] == 'High').sum()
            st.metric("High Risk Patients", high_risk)
        
        # Risk distribution chart
        self.display_risk_distribution(results_df)
        
        # Download results
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name="malaria_predictions.csv",
            mime="text/csv"
        )
    
    def display_risk_gauge(self, risk_score: float):
        """
        Display risk gauge visualization.
        
        Args:
            risk_score: Malaria risk score (0-1)
        """
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = risk_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Malaria Risk Score"},
            delta = {'reference': 0.5},
            gauge = {
                'axis': {'range': [0, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 0.3], 'color': "lightgreen"},
                    {'range': [0.3, 0.7], 'color': "yellow"},
                    {'range': [0.7, 1], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.7
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    def display_risk_distribution(self, results_df: pd.DataFrame):
        """
        Display risk distribution chart for batch predictions.
        
        Args:
            results_df: DataFrame with prediction results
        """
        risk_counts = results_df['Risk_Level'].value_counts()
        
        fig = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title="Malaria Risk Distribution",
            color=risk_counts.index,
            color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def display_feature_importance(self):
        """
        Display feature importance visualization.
        """
        if self.feature_importance is None:
            return
        
        st.markdown("### Feature Importance")
        
        # Get top 10 features
        top_features = self.feature_importance.head(10)
        
        fig = px.bar(
            top_features,
            x=top_features.iloc[:, 0],
            y=top_features.index,
            orientation='h',
            title="Top 10 Most Important Features",
            labels={'x': 'Importance', 'y': 'Feature'}
        )
        
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    def display_model_performance(self):
        """
        Display model performance metrics.
        """
        if not self.model_metrics:
            return
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Model Performance")
        
        metrics_to_display = {
            'r2': 'R² Score',
            'rmse': 'RMSE', 
            'mae': 'MAE',
            'mse': 'MSE'
        }
        
        for metric_key, display_name in metrics_to_display.items():
            if metric_key in self.model_metrics:
                value = self.model_metrics[metric_key]
                st.sidebar.metric(display_name, f"{value:.4f}")
    
    def run(self):
        """
        Run the Streamlit application.
        """
        # Page configuration
        st.set_page_config(
            page_title="Malaria Prediction System",
            page_icon="🦟",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Main title
        st.title("🦟 AI-Powered Malaria Prediction System")
        st.markdown("""
        This application uses machine learning to predict malaria risk based on patient health indicators.
        Upload individual patient data or batch process multiple records.
        """)
        
        # Load model
        with st.spinner("Loading malaria prediction model..."):
            if not self.load_model():
                st.stop()
        
        # Get user configuration
        config = self.render_sidebar()
        
        # Display model performance in sidebar
        self.display_model_performance()
        
        # Render appropriate input method
        input_data = None
        
        if config["input_method"] == "Single Prediction":
            input_data = self.render_single_prediction()
        else:
            input_data = self.render_batch_prediction()
        
        # Make predictions if data is available
        if input_data is not None and not input_data.empty:
            try:
                with st.spinner("Making predictions..."):
                    predictions = self.make_prediction(input_data)
                
                # Display results
                if config["input_method"] == "Single Prediction":
                    self.display_single_prediction_result(predictions[0], input_data)
                else:
                    self.display_batch_prediction_results(predictions, input_data)
                    
            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")
        
        # Footer
        st.markdown("---")
        st.markdown(
            "Built with ❤️ using Streamlit and Scikit-Learn | "
            "Malaria Prediction AI System"
        )


def main():
    """
    Main application entry point.
    """
    try:
        app = MalariaPredictionApp()
        app.run()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.stop()


if __name__ == "__main__":
    main()