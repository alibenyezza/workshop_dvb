"""
Baseline training script for Titanic ML workshop.

TODO: Complete this script to train a Titanic survival classifier.

Usage (after completion):
    python train.py --model logreg
    python train.py --model rf
"""

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


def load_data():
    """
    Load and prepare the Titanic dataset.
    
    TODO:
    1. Load the CSV file from data/train.csv
    2. Handle missing values (Age, Embarked, Fare)
    3. Encode categorical variables (Sex, Embarked)
    4. Select features: Pclass, Sex_encoded, Age, SibSp, Parch, Fare, Embarked_encoded
    5. Return X (features) and y (target: Survived)
    """
    # TODO: Implement data loading and preprocessing
    # Hint: Use Path(__file__).parent.parent.parent / "data" / "train.csv"
    # Hint: Use LabelEncoder for categorical variables
    # Hint: Use fillna() for missing values
    
    pass


def get_model(model_name):
    """
    Get a model instance by name.
    
    TODO:
    Support these models:
    - "logreg": LogisticRegression
    - "rf": RandomForestClassifier (n_estimators=100)
    - "gbdt": GradientBoostingClassifier (n_estimators=100)
    - "svm": SVC (probability=True)
    - "knn": KNeighborsClassifier (n_neighbors=5)
    - "extratrees": ExtraTreesClassifier (n_estimators=100)
    
    Raise ValueError if model_name is not supported.
    """
    # TODO: Implement model selection
    # Hint: Use a dictionary to map model names to model instances
    
    pass


def train_model(model_name):
    """
    Train a model and save artifacts.
    
    TODO:
    1. Load and prepare data using load_data()
    2. Split data into train/test (80/20, random_state=42, stratify=y)
    3. Get model using get_model()
    4. Train the model
    5. Evaluate on test set (accuracy and F1 score)
    6. Save model to model.pkl
    7. Save metrics to metrics.json (model, accuracy, f1)
    8. Create report.md with template
    9. Print results
    
    Output files should be saved in submissions/baseline/:
    - model.pkl
    - metrics.json
    - report.md
    """
    output_dir = Path(__file__).parent
    
    # TODO: Load data
    # X, y = load_data()
    
    # TODO: Split data
    # X_train, X_test, y_train, y_test = train_test_split(...)
    
    # TODO: Get and train model
    # model = get_model(model_name)
    # model.fit(...)
    
    # TODO: Evaluate
    # y_pred = model.predict(...)
    # accuracy = accuracy_score(...)
    # f1 = f1_score(...)
    
    # TODO: Save model
    # with open(output_dir / "model.pkl", "wb") as f:
    #     pickle.dump(model, f)
    
    # TODO: Save metrics
    # metrics = {"model": model_name, "accuracy": ..., "f1": ...}
    # with open(output_dir / "metrics.json", "w") as f:
    #     json.dump(metrics, f, indent=2)
    
    # TODO: Create report.md
    # report_content = f"""# Model Training Report
    # ...
    # """
    # with open(output_dir / "report.md", "w") as f:
    #     f.write(report_content)
    
    # TODO: Print results
    # print(f"✓ Model trained: {model_name}")
    # print(f"✓ Accuracy: {accuracy:.4f}")
    # print(f"✓ F1 Score: {f1:.4f}")
    
    pass


def main():
    """
    Main function to parse arguments and train model.
    
    TODO (for lvl3):
    - Add --seed argument (default 42)
    - Add --output-dir argument (optional)
    - Pass seed to train_model and use it for train_test_split and models
    - Save run history to runs.jsonl
    """
    parser = argparse.ArgumentParser(description="Train a Titanic survival classifier")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["logreg", "rf", "gbdt", "svm", "knn", "extratrees"],
        help="Model to train",
    )
    
    # TODO (lvl3): Add --seed argument with default 42
    # TODO (lvl3): Add --output-dir argument (optional, default None)
    
    args = parser.parse_args()
    
    # TODO: Call train_model with appropriate arguments
    # train_model(args.model)
    
    pass


if __name__ == "__main__":
    main()
