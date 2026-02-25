"""
Baseline training script for Titanic ML workshop.

Usage:
    python train.py --model logreg
    python train.py --model rf --seed 42
    python train.py --model gbdt --output-dir submissions/my-run
"""

import argparse
import json
import os
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
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
    """Load and prepare the Titanic dataset."""
    data_path = Path(__file__).parent.parent.parent / "data" / "train.csv"
    df = pd.read_csv(data_path)

    # Basic feature engineering
    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
    df["Fare"].fillna(df["Fare"].median(), inplace=True)

    # Encode categorical variables
    le_sex = LabelEncoder()
    le_embarked = LabelEncoder()
    df["Sex_encoded"] = le_sex.fit_transform(df["Sex"])
    df["Embarked_encoded"] = le_embarked.fit_transform(df["Embarked"])

    # Select features
    features = ["Pclass", "Sex_encoded", "Age", "SibSp", "Parch", "Fare", "Embarked_encoded"]
    X = df[features].values
    y = df["Survived"].values

    return X, y, le_sex, le_embarked


def get_model(model_name, random_state=42):
    """Get a model instance by name."""
    models = {
        "logreg": LogisticRegression(random_state=random_state, max_iter=1000),
        "rf": RandomForestClassifier(n_estimators=100, random_state=random_state),
        "gbdt": GradientBoostingClassifier(n_estimators=100, random_state=random_state),
        "svm": SVC(random_state=random_state, probability=True),
        "knn": KNeighborsClassifier(n_neighbors=5),
        "extratrees": ExtraTreesClassifier(n_estimators=100, random_state=random_state),
    }

    if model_name not in models:
        raise ValueError(
            f"Unknown model: {model_name}. Choose from: {list(models.keys())}"
        )

    return models[model_name]


def train_model(model_name, seed=42, output_dir=None):
    """Train a model and save artifacts."""
    if output_dir is None:
        output_dir = Path(__file__).parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    X, y, le_sex, le_embarked = load_data()

    # Split data (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    # Get and train model
    model = get_model(model_name, random_state=seed)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Save model
    model_path = output_dir / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # Save metrics
    metrics = {
        "model": model_name,
        "accuracy": float(accuracy),
        "f1": float(f1),
        "seed": seed,
        "timestamp": datetime.now().isoformat(),
    }
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save run history (append to runs.jsonl)
    runs_path = output_dir / "runs.jsonl"
    with open(runs_path, "a") as f:
        f.write(json.dumps(metrics) + "\n")

    # Create report template
    report_path = output_dir / "report.md"
    if not report_path.exists():
        report_content = f"""# Model Training Report

## Model: {model_name}

### Metrics
- **Accuracy**: {accuracy:.4f}
- **F1 Score**: {f1:.4f}
- **Seed**: {seed}

### Analysis
- [Add your analysis here]
- [Add your analysis here]

### Model Details
- Model type: {model_name}
- Training date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        with open(report_path, "w") as f:
            f.write(report_content)
    else:
        # Update existing report
        with open(report_path, "r") as f:
            content = f.read()
        # Simple update - replace metrics section
        lines = content.split("\n")
        new_lines = []
        skip_until_analysis = False
        for i, line in enumerate(lines):
            if "## Model:" in line:
                new_lines.append(f"## Model: {model_name}")
                skip_until_analysis = True
            elif "### Metrics" in line:
                new_lines.append("### Metrics")
                new_lines.append(f"- **Accuracy**: {accuracy:.4f}")
                new_lines.append(f"- **F1 Score**: {f1:.4f}")
                new_lines.append(f"- **Seed**: {seed}")
                skip_until_analysis = True
            elif "### Analysis" in line:
                skip_until_analysis = False
                new_lines.append(line)
            elif not skip_until_analysis or line.startswith("-") or line.startswith("#"):
                new_lines.append(line)

        with open(report_path, "w") as f:
            f.write("\n".join(new_lines))

    print(f"✓ Model trained: {model_name}")
    print(f"✓ Accuracy: {accuracy:.4f}")
    print(f"✓ F1 Score: {f1:.4f}")
    print(f"✓ Artifacts saved to: {output_dir}")
    print(f"  - model.pkl")
    print(f"  - metrics.json")
    print(f"  - report.md")
    print(f"  - runs.jsonl")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train a Titanic survival classifier")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["logreg", "rf", "gbdt", "svm", "knn", "extratrees"],
        help="Model to train",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for artifacts (default: submissions/baseline)",
    )

    args = parser.parse_args()

    # Set random seed for reproducibility
    np.random.seed(args.seed)

    train_model(args.model, seed=args.seed, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
