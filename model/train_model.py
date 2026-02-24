"""
=============================================================
 Decision Tree Model Trainer for DNIDS
=============================================================
 This script:
 1. Loads the KDD Cup 99 dataset (10% version)
 2. Preprocesses all 41 features:
    - Encodes categorical features (protocol_type, service, flag)
    - Scales numerical features using StandardScaler
 3. Converts labels to binary: normal=0, attack=1
 4. Trains a DecisionTreeClassifier
 5. Evaluates the model (accuracy, precision, recall, F1)
 6. Saves the trained model and preprocessors as .pkl files

 Output Files:
   - model/decision_tree_model.pkl   (trained classifier)
   - model/scaler.pkl                (fitted StandardScaler)
   - model/label_encoders.pkl        (fitted LabelEncoders)

 Usage:
   python model/train_model.py
=============================================================
"""

import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# ─────────────────────────────────────────────
# Add parent directory to path for imports
# ─────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from dataset.download_dataset import (
    get_dataset_path, get_feature_names, get_all_columns, download_dataset
)

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, "decision_tree_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
ENCODERS_PATH = os.path.join(MODEL_DIR, "label_encoders.pkl")

# Categorical feature columns (these need label encoding)
CATEGORICAL_FEATURES = ['protocol_type', 'service', 'flag']

# Random seed for reproducibility
RANDOM_SEED = 42

# Test split ratio
TEST_SIZE = 0.3


def load_dataset():
    """
    Load the KDD Cup 99 dataset into a pandas DataFrame.
    
    Returns:
        pd.DataFrame: The loaded dataset with proper column names.
    """
    print("\n📂 Loading KDD Cup 99 dataset...")
    
    # Get dataset path (download if needed)
    csv_path = get_dataset_path()
    if not os.path.exists(csv_path):
        print("   Dataset not found. Downloading...")
        download_dataset()
    
    # Load CSV without headers (the KDD dataset has no header row)
    columns = get_all_columns()
    df = pd.read_csv(csv_path, header=None, names=columns)
    
    print(f"   ✅ Loaded {len(df):,} records with {len(df.columns)} columns")
    print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
    
    return df


def explore_dataset(df):
    """
    Print exploratory statistics about the dataset.
    
    Args:
        df: pandas DataFrame of the dataset.
    """
    print(f"\n{'─' * 60}")
    print(f"  📊 Dataset Exploration")
    print(f"{'─' * 60}")
    
    # Shape
    print(f"\n  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # Data types
    print(f"\n  Data Types:")
    print(f"    Numerical:   {len(df.select_dtypes(include=[np.number]).columns)} columns")
    print(f"    Categorical: {len(df.select_dtypes(include=['object']).columns)} columns")
    
    # Label distribution
    print(f"\n  Label Distribution:")
    label_counts = df['label'].value_counts()
    total = len(df)
    for label, count in label_counts.head(10).items():
        pct = count / total * 100
        bar = '█' * int(pct / 2)
        print(f"    {label:<25s} {count:>8,} ({pct:>5.1f}%) {bar}")
    if len(label_counts) > 10:
        print(f"    ... and {len(label_counts) - 10} more categories")
    
    # Binary classification breakdown
    normal_count = (df['label'] == 'normal.').sum()
    attack_count = (df['label'] != 'normal.').sum()
    print(f"\n  Binary Classification:")
    print(f"    Normal (0): {normal_count:>8,} ({normal_count/total*100:.1f}%)")
    print(f"    Attack (1): {attack_count:>8,} ({attack_count/total*100:.1f}%)")
    
    # Categorical feature values
    print(f"\n  Categorical Feature Values:")
    for col in CATEGORICAL_FEATURES:
        unique = df[col].nunique()
        values = df[col].unique()[:5]
        print(f"    {col}: {unique} unique → {list(values)}{'...' if unique > 5 else ''}")


def preprocess_data(df):
    """
    Preprocess the KDD Cup 99 dataset for training.
    
    Steps:
    1. Encode categorical features using LabelEncoder
    2. Convert labels to binary (normal=0, attack=1)
    3. Split into features (X) and labels (y)
    4. Scale features using StandardScaler
    
    Args:
        df: pandas DataFrame of the dataset.
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler, label_encoders)
    """
    print(f"\n{'─' * 60}")
    print(f"  ⚙️  Preprocessing Data")
    print(f"{'─' * 60}")
    
    # Step 1: Encode categorical features
    print(f"\n  Step 1: Encoding categorical features...")
    label_encoders = {}
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        print(f"    ✅ {col}: {len(le.classes_)} classes encoded")
    
    # Step 2: Convert labels to binary
    print(f"\n  Step 2: Converting labels to binary...")
    # normal. → 0, everything else → 1
    df['binary_label'] = (df['label'] != 'normal.').astype(int)
    print(f"    ✅ normal. → 0, all attacks → 1")
    
    # Step 3: Separate features and labels
    print(f"\n  Step 3: Separating features and labels...")
    feature_names = get_feature_names()
    X = df[feature_names].values
    y = df['binary_label'].values
    print(f"    ✅ Features shape: {X.shape}")
    print(f"    ✅ Labels shape: {y.shape}")
    
    # Step 4: Train/Test split
    print(f"\n  Step 4: Splitting data (train: {(1-TEST_SIZE)*100:.0f}%, test: {TEST_SIZE*100:.0f}%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    print(f"    ✅ Training set: {X_train.shape[0]:,} samples")
    print(f"    ✅ Test set:     {X_test.shape[0]:,} samples")
    
    # Step 5: Scale features
    print(f"\n  Step 5: Scaling features with StandardScaler...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print(f"    ✅ Features scaled (mean≈0, std≈1)")
    
    return X_train, X_test, y_train, y_test, scaler, label_encoders


def train_model(X_train, y_train):
    """
    Train a DecisionTreeClassifier.
    
    Args:
        X_train: Training feature matrix.
        y_train: Training labels.
    
    Returns:
        DecisionTreeClassifier: Trained model.
    """
    print(f"\n{'─' * 60}")
    print(f"  🤖 Training Decision Tree Classifier")
    print(f"{'─' * 60}")
    
    # Create the model
    model = DecisionTreeClassifier(
        criterion='gini',           # Split quality measure
        max_depth=20,               # Maximum tree depth (prevent overfitting)
        min_samples_split=10,       # Minimum samples to split a node
        min_samples_leaf=5,         # Minimum samples in a leaf
        random_state=RANDOM_SEED,   # Reproducibility
        class_weight='balanced'     # Handle class imbalance
    )
    
    print(f"\n  Model Parameters:")
    print(f"    Algorithm:        Decision Tree (CART)")
    print(f"    Criterion:        Gini Impurity")
    print(f"    Max Depth:        {model.max_depth}")
    print(f"    Min Samples Split:{model.min_samples_split}")
    print(f"    Min Samples Leaf: {model.min_samples_leaf}")
    print(f"    Class Weight:     Balanced")
    
    # Train
    print(f"\n  Training...")
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    print(f"    ✅ Training completed in {train_time:.2f} seconds")
    print(f"    Tree depth: {model.get_depth()}")
    print(f"    Number of leaves: {model.get_n_leaves()}")
    
    # Feature importance (top 10)
    feature_names = get_feature_names()
    importances = model.feature_importances_
    top_indices = np.argsort(importances)[::-1][:10]
    
    print(f"\n  Top 10 Most Important Features:")
    for rank, idx in enumerate(top_indices, 1):
        bar = '█' * int(importances[idx] * 50)
        print(f"    {rank:>2}. {feature_names[idx]:<30s} {importances[idx]:.4f} {bar}")
    
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on the test set.
    
    Args:
        model: Trained DecisionTreeClassifier.
        X_test: Test feature matrix.
        y_test: Test labels.
    """
    print(f"\n{'─' * 60}")
    print(f"  📈 Model Evaluation")
    print(f"{'─' * 60}")
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')
    
    print(f"\n  Performance Metrics:")
    print(f"    ┌─────────────────────────────────┐")
    print(f"    │ Accuracy:  {accuracy*100:>6.2f}%             │")
    print(f"    │ Precision: {precision*100:>6.2f}%             │")
    print(f"    │ Recall:    {recall*100:>6.2f}%             │")
    print(f"    │ F1 Score:  {f1*100:>6.2f}%             │")
    print(f"    └─────────────────────────────────┘")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"                    Predicted")
    print(f"                 Normal   Attack")
    print(f"    Actual Normal  {cm[0][0]:>7,}  {cm[0][1]:>7,}")
    print(f"    Actual Attack  {cm[1][0]:>7,}  {cm[1][1]:>7,}")
    
    # Classification Report
    print(f"\n  Detailed Classification Report:")
    target_names = ['Normal (0)', 'Attack (1)']
    report = classification_report(y_test, y_pred, target_names=target_names)
    for line in report.split('\n'):
        print(f"    {line}")
    
    return accuracy


def save_model(model, scaler, label_encoders, accuracy):
    """
    Save the trained model and preprocessors as .pkl files.
    
    Args:
        model: Trained DecisionTreeClassifier.
        scaler: Fitted StandardScaler.
        label_encoders: Dict of fitted LabelEncoders.
        accuracy: Model accuracy for metadata.
    """
    print(f"\n{'─' * 60}")
    print(f"  💾 Saving Model & Preprocessors")
    print(f"{'─' * 60}")
    
    # Save model
    joblib.dump(model, MODEL_PATH)
    model_size = os.path.getsize(MODEL_PATH) / 1024
    print(f"\n  ✅ Model saved: {MODEL_PATH}")
    print(f"     Size: {model_size:.1f} KB")
    
    # Save scaler
    joblib.dump(scaler, SCALER_PATH)
    scaler_size = os.path.getsize(SCALER_PATH) / 1024
    print(f"  ✅ Scaler saved: {SCALER_PATH}")
    print(f"     Size: {scaler_size:.1f} KB")
    
    # Save label encoders
    joblib.dump(label_encoders, ENCODERS_PATH)
    encoders_size = os.path.getsize(ENCODERS_PATH) / 1024
    print(f"  ✅ Encoders saved: {ENCODERS_PATH}")
    print(f"     Size: {encoders_size:.1f} KB")
    
    # Save metadata
    metadata = {
        'accuracy': accuracy,
        'model_type': 'DecisionTreeClassifier',
        'features': 41,
        'categorical_features': CATEGORICAL_FEATURES,
        'binary_classification': True,
        'labels': {0: 'Normal', 1: 'Attack'},
    }
    metadata_path = os.path.join(MODEL_DIR, "model_metadata.txt")
    with open(metadata_path, 'w') as f:
        f.write("=" * 50 + "\n")
        f.write("  DNIDS Model Metadata\n")
        f.write("=" * 50 + "\n\n")
        for key, value in metadata.items():
            f.write(f"  {key}: {value}\n")
    print(f"  ✅ Metadata saved: {metadata_path}")
    
    print(f"\n  Total saved: {(model_size + scaler_size + encoders_size):.1f} KB")


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("  🛡️  DNIDS - Model Training Pipeline")
    print("=" * 60)
    
    overall_start = time.time()
    
    # Step 1: Load dataset
    df = load_dataset()
    
    # Step 2: Explore dataset
    explore_dataset(df)
    
    # Step 3: Preprocess data
    X_train, X_test, y_train, y_test, scaler, label_encoders = preprocess_data(df)
    
    # Step 4: Train model
    model = train_model(X_train, y_train)
    
    # Step 5: Evaluate model
    accuracy = evaluate_model(model, X_test, y_test)
    
    # Step 6: Save everything
    save_model(model, scaler, label_encoders, accuracy)
    
    # Summary
    total_time = time.time() - overall_start
    print(f"\n{'=' * 60}")
    print(f"  ✅ Training Pipeline Complete!")
    print(f"  Total time: {total_time:.1f} seconds")
    print(f"  Model accuracy: {accuracy*100:.2f}%")
    print(f"{'=' * 60}")


# ─────────────────────────────────────────────
# Main Execution
# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()
