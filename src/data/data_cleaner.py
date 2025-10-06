"""
CLEAN DATA CLEANER - Fixed Version (No Warnings)
"""
import pandas as pd
import numpy as np
import os

def load_dataset():
    """Find and load the dataset"""
    print("🔍 Loading DatasetAfricaMalaria.csv...")
    
    paths = [
        "../data/DatasetAfricaMalaria.csv",
        "DatasetAfricaMalaria.csv",  
        "../../data/DatasetAfricaMalaria.csv",
    ]
    
    for path in paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"✅ Loaded: {path} ({df.shape[0]} rows, {df.shape[1]} cols)")
            return df
    
    print("❌ Dataset not found.")
    return None

def clean_column_names(df):
    """Simple column standardization"""
    print("🔄 Cleaning column names...")
    df.columns = [str(col).strip().lower().replace(' ', '_').replace('-', '_') 
                 for col in df.columns]
    return df

def handle_missing_values(df, threshold=0.5):
    """Smart missing value handling - FIXED no warnings"""
    print("🔍 Handling missing values...")
    
    # Drop columns with too many missing values
    missing_ratio = df.isnull().sum() / len(df)
    cols_to_drop = missing_ratio[missing_ratio > threshold].index
    df = df.drop(columns=cols_to_drop)
    
    if len(cols_to_drop) > 0:
        print(f"🗑️ Dropped {len(cols_to_drop)} columns with >{threshold*100}% missing")
    
    # FIXED: No chained assignment
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['int64', 'float64']:
                # Fixed: Assign directly instead of inplace
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna('Unknown')
    
    return df

def create_target(df):
    """Create malaria prediction target"""
    print("🎯 Creating target variable...")
    
    malaria_cols = [col for col in df.columns if 'malaria' in col.lower()]
    
    if malaria_cols:
        target_col = malaria_cols[0]
        print(f"🎯 Using '{target_col}' as target")
        
        if df[target_col].dtype in ['int64', 'float64']:
            threshold = df[target_col].median()
            df['malaria_risk_high'] = (df[target_col] > threshold).astype(int)
            print(f"📊 Created binary target (threshold: {threshold:.2f})")
            print(f"📈 Class balance: {df['malaria_risk_high'].value_counts().to_dict()}")
    
    return df

def validate_data(df):
    """Quick data validation"""
    print("🔍 Validating data...")
    
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        df = df.drop_duplicates()
        print(f"🗑️ Removed {duplicates} duplicates")
    
    print(f"📊 Final shape: {df.shape}")
    print(f"🔢 Numeric columns: {len(df.select_dtypes(include=[np.number]).columns)}")
    print(f"📝 Categorical columns: {len(df.select_dtypes(include=['object']).columns)}")
    
    return df

def save_cleaned_data(df):
    """Save the cleaned dataset"""
    output_path = "../data/cleaned_malaria_data.csv"
    df.to_csv(output_path, index=False)
    print(f"💾 Saved cleaned data to: {output_path}")
    return output_path

def main():
    """Simple cleaning pipeline"""
    print("🚀 Starting Malaria Data Cleaning...")
    
    df = load_dataset()
    if df is None:
        return
    
    df = clean_column_names(df)
    df = handle_missing_values(df)
    df = create_target(df) 
    df = validate_data(df)
    output_path = save_cleaned_data(df)
    
    print(f"\n✅ Cleaning completed!")
    print(f"📁 Output: {output_path}")
    print(f"🎯 Next: Run model training on cleaned data")

if __name__ == "__main__":
    main()