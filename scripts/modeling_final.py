import pandas as pd
import numpy as np
import ast
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --- Configuration ---
RAW_DATA_PATH = 'data/processed/integrated_tv_shows.csv'
TRANSFORMED_DATA_PATH = 'data/processed/transformed_tv_shows.csv'
REPORT_PATH = 'data/processed/modeling_report.txt'
TARGET_GENRE = 'Comedy'
RANDOM_STATE = 42

def parse_genres(x):
    try:
        return ast.literal_eval(x)
    except:
        return []

def load_and_prep_raw():
    print("Loading Raw Data...")
    df = pd.read_csv(RAW_DATA_PATH)
    
    # Target Extraction
    df['genres_list'] = df['genres'].apply(parse_genres)
    df['target'] = df['genres_list'].apply(lambda x: 1 if TARGET_GENRE in x else 0)
    
    # Feature Selection (Raw features that might be useful)
    # We avoid text descriptions for this basic modeling, focusing on metadata
    features = ['release_year', 'runtime_minutes', 'language', 'type', 'status']
    
    X = df[features]
    y = df['target']
    
    # Preprocessing Pipeline for Raw Data
    # - Numeric: Impute missing with median
    # - Categorical: Impute missing with 'Unknown', then OneHotEncode
    
    numeric_features = ['release_year', 'runtime_minutes']
    categorical_features = ['language', 'type', 'status']
    
    numeric_transformer = SimpleImputer(strategy='median')
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
        
    return X, y, preprocessor

def load_and_prep_transformed():
    print("Loading Transformed Data...")
    df = pd.read_csv(TRANSFORMED_DATA_PATH)
    
    # The transformed data might not have the 'target' column explicitly if it wasn't preserved.
    # However, it usually has 'genres_parsed'. Let's check.
    # If 'genres_parsed' is there, we use it. If not, we might need to join with raw.
    # Based on previous steps, 'genres_parsed' should be there.
    
    if 'genres_parsed' in df.columns:
        df['genres_list'] = df['genres_parsed'].apply(parse_genres)
        df['target'] = df['genres_list'].apply(lambda x: 1 if TARGET_GENRE in x else 0)
    else:
        # Fallback: Join with raw on ID to get target
        print("Warning: 'genres_parsed' not found in transformed data. Joining with raw data to get target.")
        df_raw = pd.read_csv(RAW_DATA_PATH)[['id', 'genres']]
        df = df.merge(df_raw, on='id', how='left')
        df['genres_list'] = df['genres'].apply(parse_genres)
        df['target'] = df['genres_list'].apply(lambda x: 1 if TARGET_GENRE in x else 0)

    # Features from transformed schema
    # We use the scaled/encoded features
    feature_cols = [c for c in df.columns if c not in ['id', 'title_normalized', 'genres_parsed', 'genres', 'genres_list', 'target', 'Unnamed: 0']]
    
    # Filter out any object columns that might have slipped through or need encoding
    # Ideally transformed data is all numeric, but 'language', 'type', 'status' might be there as text if not fully encoded in transformation step.
    # Let's check dtypes.
    
    X = df[feature_cols]
    y = df['target']
    
    # Identify categorical vs numeric in X
    # If transformation step already encoded everything, this will be all numeric.
    # If not, we apply encoding.
    
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    numeric_transformer = SimpleImputer(strategy='median') # Just in case
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return X, y, preprocessor

def train_and_evaluate(X, y, preprocessor, dataset_name, report_file):
    print(f"\n--- Processing {dataset_name} Data ---")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE)
    }
    
    report_file.write(f"\n{'='*20}\nDataset: {dataset_name}\n{'='*20}\n")
    
    results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Create full pipeline
        clf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', model)])
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        results[name] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1}
        
        report_file.write(f"\nModel: {name}\n")
        report_file.write(f"Accuracy:  {acc:.4f}\n")
        report_file.write(f"Precision: {prec:.4f}\n")
        report_file.write(f"Recall:    {rec:.4f}\n")
        report_file.write(f"F1 Score:  {f1:.4f}\n")
        report_file.write("-" * 20 + "\n")
        
    return results

def main():
    with open(REPORT_PATH, 'w') as f:
        f.write("TV Show Project - Modeling Report\n")
        f.write(f"Target Variable: Is '{TARGET_GENRE}'?\n")
        f.write("=================================\n")
        
        # 1. Raw Data
        X_raw, y_raw, prep_raw = load_and_prep_raw()
        train_and_evaluate(X_raw, y_raw, prep_raw, "RAW", f)
        
        # 2. Transformed Data
        X_trans, y_trans, prep_trans = load_and_prep_transformed()
        train_and_evaluate(X_trans, y_trans, prep_trans, "TRANSFORMED", f)
        
    print(f"\nModeling complete. Report saved to {REPORT_PATH}")

if __name__ == "__main__":
    main()
