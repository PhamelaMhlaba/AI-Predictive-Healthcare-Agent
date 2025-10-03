import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def explore_dataset():
    """Quick and dirty data exploration - let's see what we're working with"""
    print("🔍 EXPLORING YOUR DATASETAFRICAMALARIA")
    print("=" * 60)
    
    # Load the data
    try:
        # Try different possible file locations
        if os.path.exists("data/DatasetAfricaMalaria"):
            df = pd.read_csv("data/DatasetAfricaMalaria")
            print("✅ Loaded: data/DatasetAfricaMalaria")
        elif os.path.exists("data/DatasetAfricaMalaria.csv"):
            df = pd.read_csv("data/DatasetAfricaMalaria.csv")
            print("✅ Loaded: data/DatasetAfricaMalaria.csv")
        else:
            print("❌ File not found!")
            return
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return
    
    # BASIC DATASET INFO
    print(f"\n📊 DATASET SHAPE: {df.shape}")
    print(f"📋 COLUMNS: {df.columns.tolist()}")
    
    print(f"\n📈 DATA TYPES:")
    print(df.dtypes)
    
    print(f"\n❓ MISSING VALUES:")
    missing = df.isnull().sum()
    print(missing[missing > 0])
    
    # BASIC STATISTICS
    print(f"\n📊 BASIC STATISTICS:")
    print(df.describe())
    
    # FIRST FEW ROWS
    print(f"\n👀 FIRST 5 ROWS:")
    print(df.head())
    
    # CHECK FOR MALARIA-RELATED COLUMNS
    malaria_keywords = ['malaria', 'case', 'death', 'incidence', 'prevalence', 'risk', 'fever']
    malaria_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in malaria_keywords)]
    
    print(f"\n🎯 MALARIA-RELATED COLUMNS FOUND: {malaria_columns}")
    
    if malaria_columns:
        print(f"\n📊 MALARIA COLUMNS INFO:")
        for col in malaria_columns:
            print(f"{col}:")
            print(f"  - Unique values: {df[col].nunique()}")
            print(f"  - Sample values: {df[col].unique()[:5]}")
            if df[col].dtype in ['int64', 'float64']:
                print(f"  - Range: {df[col].min()} to {df[col].max()}")
    
    # CHECK FOR GEOGRAPHIC COLUMNS
    geo_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in ['country', 'region', 'district', 'zone', 'area'])]
    print(f"\n🌍 GEOGRAPHIC COLUMNS: {geo_columns}")
    
    # CHECK FOR RISK FACTOR COLUMNS
    risk_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in 
                   ['rain', 'temp', 'humid', 'population', 'density', 'poverty', 'health', 'sanitation'])]
    print(f"\n📈 RISK FACTOR COLUMNS: {risk_columns}")
    
    return df

def create_quick_visualizations(df):
    """Create some quick plots to understand the data"""
    print(f"\n📊 CREATING QUICK VISUALIZATIONS...")
    
    # Set up plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Check for numeric columns distribution
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) > 0:
        # Plot first 4 numeric columns
        for i, col in enumerate(numeric_cols[:4]):
            ax = axes[i//2, i%2]
            df[col].hist(ax=ax, bins=20, alpha=0.7)
            ax.set_title(f'Distribution of {col}')
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('data_exploration.png', dpi=100, bbox_inches='tight')
    print("✅ Saved visualization as 'data_exploration.png'")
    
    # 2. Correlation heatmap if we have enough numeric columns
    if len(numeric_cols) > 2:
        plt.figure(figsize=(12, 8))
        correlation_matrix = df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix of Numeric Features')
        plt.tight_layout()
        plt.savefig('correlation_matrix.png', dpi=100, bbox_inches='tight')
        print("✅ Saved correlation matrix as 'correlation_matrix.png'")

def check_data_quality(df):
    """Check data quality issues"""
    print(f"\n🔍 DATA QUALITY CHECK:")
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    print(f"🔁 Duplicate rows: {duplicates}")
    
    # Check for constant columns
    constant_cols = [col for col in df.columns if df[col].nunique() == 1]
    print(f"📏 Constant columns: {constant_cols}")
    
    # Check for high cardinality categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    high_cardinality = []
    for col in categorical_cols:
        if df[col].nunique() > 50:
            high_cardinality.append((col, df[col].nunique()))
    
    if high_cardinality:
        print(f"🎯 High cardinality categorical columns: {high_cardinality}")
        
if __name__ == "__main__":
    print("🚀 STARTING DATA EXPLORATION...")
    df = explore_dataset()
    
    if df is not None:
        create_quick_visualizations(df)
        check_data_quality(df)
        
        print(f"\n🎯 NEXT STEPS SUGGESTIONS:")
        print("1. Review the output above to understand your data")
        print("2. Check the generated PNG files for visual insights")
        print("3. Identify which column should be your target variable")
        print("4. Decide which features to use for prediction")