"""
FIXED MALARIA DATA PREPROCESSOR
Corrected version with proper feature name extraction
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
from pathlib import Path

class MalariaPreprocessor:
    """
    Robust preprocessing with dimensionality control - FIXED VERSION
    """
    
    def __init__(self, random_state=42, test_size=0.2, max_categories=10):
        self.random_state = random_state
        self.test_size = test_size
        self.max_categories = max_categories
        self.preprocessor = None
        self.numeric_features_used = []
        self.categorical_features_used = []
        
        self.PATHS = {
            'input': ['cleaned_malaria_data.csv', '../data/cleaned_malaria_data.csv'],
            'output': Path('../../data/processed')
        }

    def load_data(self) -> pd.DataFrame:
        """Load data with validation"""
        for path in self.PATHS['input']:
            if Path(path).exists():
                df = pd.read_csv(path)
                print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
                return df
        raise FileNotFoundError("Data not found")

    def analyze_and_clean_features(self, df: pd.DataFrame) -> tuple:
        """Smart feature analysis with cardinality control"""
        target_col = 'malaria_risk_high'
        
        if target_col not in df.columns:
            raise ValueError(f"Target '{target_col}' not found")
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Remove problematic columns
        if 'geometry' in X.columns:
            X = X.drop(columns=['geometry'])
        
        # Analyze feature types
        numeric_features = X.select_dtypes(include=np.number).columns.tolist()
        categorical_features = X.select_dtypes(include='object').columns.tolist()
        
        # Store for later use
        self.numeric_features_used = numeric_features.copy()
        
        # Control categorical cardinality
        safe_categorical = []
        for col in categorical_features:
            unique_count = X[col].nunique()
            if unique_count <= self.max_categories:
                safe_categorical.append(col)
            else:
                print(f"Dropping high-cardinality column: {col} ({unique_count} categories)")
                X = X.drop(columns=[col])
        
        self.categorical_features_used = safe_categorical.copy()
        
        print(f"Final: {len(numeric_features)} numeric, {len(safe_categorical)} categorical")
        print(f"Target balance: {y.value_counts().to_dict()}")
        
        return X, y, numeric_features, safe_categorical

    def create_smart_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create meaningful composite features"""
        # Healthcare access (using available columns)
        water_sanitation_cols = [col for col in X.columns if any(word in col for word in 
                                ['drinking_water', 'sanitation'])]
        if water_sanitation_cols:
            X['water_sanitation_score'] = X[water_sanitation_cols].mean(axis=1)
            print(f"Created water_sanitation_score from {water_sanitation_cols}")
        
        # Population dynamics
        pop_cols = [col for col in X.columns if 'population' in col.lower()]
        if pop_cols:
            X['population_score'] = X[pop_cols].mean(axis=1)
            print(f"Created population_score from {pop_cols}")
        
        return X

    def build_efficient_pipeline(self, numeric_features: list, categorical_features: list):
        """Build pipeline with controlled dimensionality"""
        transformers = []
        
        # Numeric pipeline
        if numeric_features:
            numeric_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            transformers.append(('num', numeric_pipeline, numeric_features))
        
        # Categorical pipeline (only if there are categorical features)
        if categorical_features:
            categorical_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'))
            ])
            transformers.append(('cat', categorical_pipeline, categorical_features))
        
        preprocessor = ColumnTransformer(transformers, remainder='drop')
        return preprocessor

    def _get_feature_names(self) -> list:
        """FIXED: Properly extract feature names after preprocessing"""
        feature_names = []
        
        # Numeric features (unchanged names)
        feature_names.extend(self.numeric_features_used)
        
        # Categorical features (one-hot encoded)
        if self.categorical_features_used:
            # Get the fitted categorical transformer
            cat_transformer = self.preprocessor.named_transformers_['cat']
            onehot_encoder = cat_transformer.named_steps['onehot']
            
            # Get one-hot encoded feature names
            onehot_features = onehot_encoder.get_feature_names_out(self.categorical_features_used)
            feature_names.extend(onehot_features)
        
        print(f"Feature names extracted: {len(feature_names)} total")
        return feature_names

    def execute(self):
        """Optimized execution pipeline - FIXED"""
        try:
            print("RUNNING FIXED PREPROCESSOR...")
            print("=" * 50)
            
            # 1. Load and prepare data
            df = self.load_data()
            X, y, numeric_features, categorical_features = self.analyze_and_clean_features(df)
            
            # 2. Feature engineering
            X = self.create_smart_features(X)
            
            # 3. Update feature lists after engineering
            numeric_features = X.select_dtypes(include=np.number).columns.tolist()
            categorical_features = X.select_dtypes(include='object').columns.tolist()
            
            # Store updated lists
            self.numeric_features_used = numeric_features
            self.categorical_features_used = categorical_features
            
            print(f"🔧 After engineering: {len(numeric_features)} numeric, {len(categorical_features)} categorical")
            
            # 4. Build and fit pipeline
            self.preprocessor = self.build_efficient_pipeline(numeric_features, categorical_features)
            X_processed = self.preprocessor.fit_transform(X)
            
            # 5. Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y, 
                test_size=self.test_size, 
                random_state=self.random_state, 
                stratify=y
            )
            
            print(f"Split: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
            
            # 6. Get feature names
            feature_names = self._get_feature_names()
            
            # 7. Save results
            self.PATHS['output'].mkdir(parents=True, exist_ok=True)
            
            # Convert to DataFrames with proper column names
            X_train_df = pd.DataFrame(X_train, columns=feature_names)
            X_test_df = pd.DataFrame(X_test, columns=feature_names)
            
            X_train_df.to_csv(self.PATHS['output'] / 'X_train.csv', index=False)
            X_test_df.to_csv(self.PATHS['output'] / 'X_test.csv', index=False)
            y_train.to_csv(self.PATHS['output'] / 'y_train.csv', index=False)
            y_test.to_csv(self.PATHS['output'] / 'y_test.csv', index=False)
            
            joblib.dump(self.preprocessor, self.PATHS['output'] / 'preprocessor.pkl')
            
            print(f"Final feature count: {len(feature_names)}")
            print(f"Saved to: {self.PATHS['output']}")
            
            return True
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Run the fixed preprocessor"""
    preprocessor = MalariaPreprocessor(max_categories=15)
    success = preprocessor.execute()
    
    if success:
        print("\nPREPROCESSING COMPLETED SUCCESSFULLY!")
        print("Ready for model training!")
    else:
        print("\nPreprocessing failed")

if __name__ == "__main__":
    main()