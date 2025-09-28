# 📊 Data Directory

## 📁 Healthcare Insurance Cost Dataset

This directory contains the healthcare insurance cost prediction dataset and related data files used in this project.

### 📈 Dataset Overview

**File**: `insurance.csv`
- **Size**: 1,338 records
- **Features**: 7 columns (6 predictors + 1 target)
- **Target Variable**: `charges` (insurance cost in USD)
- **Data Quality**: Clean dataset, no missing values
- **Source**: Healthcare insurance company records

### 🔍 Dataset Schema

| Column | Type | Description | Example Values |
|--------|------|-------------|----------------|
| `age` | Integer | Policyholder age (18-64) | 19, 27, 33, 45 |
| `sex` | Categorical | Gender (male/female) | male, female |
| `bmi` | Float | Body Mass Index (15.96-53.13) | 27.9, 33.77, 28.88 |
| `children` | Integer | Number of dependents (0-5) | 0, 1, 2, 3 |
| `smoker` | Categorical | Smoking status (yes/no) | yes, no |
| `region` | Categorical | US geographic region | southwest, southeast, northwest, northeast |
| `charges` | Float | **TARGET**: Insurance charges ($1,121-$63,770) | 16884.92, 1725.55, 4449.46 |

### 📊 Data Statistics

```
Dataset Shape: (1338, 7)
Target Variable Range: $1,121.87 - $63,770.43
Average Insurance Cost: $13,270.42
Smoker Premium Impact: ~4x higher costs
Age Distribution: 18-64 years (uniform)
Regional Distribution: Balanced across 4 US regions
```

### 🗂️ File Structure

```
data/
├── README.md              # This documentation
├── insurance.csv          # Main dataset (original)
├── processed/             # Processed data files
│   ├── train_data.csv     # Training set (80%)
│   ├── test_data.csv      # Test set (20%)
│   ├── validation_data.csv # Validation set for model tuning
│   └── feature_engineered.csv # With derived features
├── external/              # Additional datasets
│   ├── medical_costs_reference.csv
│   └── healthcare_inflation_rates.csv
└── raw/                   # Original unprocessed files
    └── insurance_raw_backup.csv
    ```

    ### 🔧 Data Preprocessing Pipeline

    1. **Initial Analysis**: Exploratory data analysis and visualization
    2. **Feature Engineering**: 
       - BMI categories (Underweight, Normal, Overweight, Obese)
          - Age groups (Young Adult, Adult, Middle-aged, Senior)
             - Premium risk factors combination
             3. **Encoding**: Label encoding for categorical variables
             4. **Scaling**: StandardScaler for numerical features
             5. **Splitting**: 80/20 train-test split with stratification

             ### 📚 Usage Examples

             ```python
             import pandas as pd
             import numpy as np

             # Load the dataset
             data = pd.read_csv('data/insurance.csv')

             # Basic exploration
             print(f"Dataset shape: {data.shape}")
             print(f"Missing values: {data.isnull().sum().sum()}")
             print(f"Target statistics:\n{data['charges'].describe()}")

             # Load processed data
             train_data = pd.read_csv('data/processed/train_data.csv')
             test_data = pd.read_csv('data/processed/test_data.csv')
             ```

             ### ⚠️ Data Usage Guidelines

             - **Ethical Use**: This dataset is for educational and research purposes
             - **Privacy**: All personal identifiers have been removed
             - **Accuracy**: Results should not be used for actual insurance pricing
             - **Citation**: Please cite the original data source when using

             ### 🔗 Data Sources

             - Primary Dataset: Healthcare Insurance Cost Dataset
             - External References: Medical cost inflation data (Bureau of Labor Statistics)
             - Validation Data: Cross-referenced with industry benchmarks

             ### 📝 Data Quality Notes

             ✅ **Strengths:**
             - No missing values
             - Balanced categorical distributions
             - Realistic value ranges
             - Good target variable distribution

             ⚠️ **Considerations:**
             - Limited geographic scope (US only)
             - Snapshot data (single time period)
             - Simplified feature set
             - No temporal trends captured

             ---

             *📧 Contact: For questions about data usage or additional datasets, please open an issue in this repository.*
