# Titanic ML Model - Git + ML Workshop

This workshop helps each participant train a simple local Titanic classifier, save model metrics, and submit their result through a GitHub Pull Request.

## 📋 Table of Contents

- [Setup](#1-setup)
- [Dataset](#2-dataset)
- [Workshop Flow](#3-workshop-flow)
- [Exercises by Level](#4-exercises-by-level)
- [Git Workflow](#5-git-workflow-branch--pr)
- [Rules](#6-rules)
- [Workshop Objective](#7-workshop-objective)

## 1) Setup

Python version: **3.10+**

```bash
python -m venv .venv
```

### Windows (PowerShell)
```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### macOS / Linux
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Dataset

This workshop uses the Titanic dataset from the Kaggle competition **Titanic - Machine Learning from Disaster**.

- Competition link: https://www.kaggle.com/competitions/titanic
- Goal: predict passenger survival (`Survived`) from passenger features (age, sex, class, fare, embarked port, etc.)
- Local files used in this workshop:
  - `data/train.csv` for training/evaluation
  - `data/test.csv` and `data/gender_submission.csv` are provided for reference

## 3) Workshop Flow

1. Train the baseline model:
```bash
python submissions/baseline/train.py --model logreg
```

2. Try other models:
```bash
python submissions/baseline/train.py --model rf
python submissions/baseline/train.py --model gbdt
python submissions/baseline/train.py --model svm
python submissions/baseline/train.py --model knn
python submissions/baseline/train.py --model extratrees
```

3. Check generated files in `submissions/baseline/`:
- `model.pkl`
- `metrics.json`
- `report.md`

## 4) Exercises by Level

### lvl1 - First Clean Run

Goal: run one full training and open a valid PR.

1. Run:
```bash
python submissions/baseline/train.py --model logreg
```
2. Confirm generated files:
   - `submissions/baseline/model.pkl`
   - `submissions/baseline/metrics.json`
   - `submissions/baseline/report.md`
3. Update `report.md` with:
   - model used
   - accuracy
   - f1
   - 2 short analysis bullets
4. Create branch, commit, open PR.

Success criteria:
- script runs without error
- artifacts are present
- PR is clean and readable

### lvl2 - Compare Multiple Models

Goal: perform a mini benchmark and justify a model choice.

1. Test at least 4 models among:
   - `logreg`, `rf`, `gbdt`, `svm`, `knn`, `extratrees`
2. Track accuracy and f1 for each run.
3. Add a comparison table in `report.md`.
4. Keep the best run (by f1) in `metrics.json`.
5. Explain in 3-5 lines why this model is selected.

Success criteria:
- factual comparison
- f1-based final choice
- reproducibility preserved (no `src/` edits, unchanged split)

### lvl3 - Improve Training Workflow

Goal: make submission workflow more robust without advanced ML changes.

1. Extend `submissions/baseline/train.py` with:
   - `--seed` (default `42`)
   - `--output-dir` (custom output folder)
2. Ensure supported models use the provided seed.
3. Save run history to `runs.jsonl`:
   - one JSON line per run with timestamp, model, accuracy, f1, seed
4. Make the script robust:
   - create output folder if missing
   - keep repeated runs predictable
5. Update `report.md` with advanced usage instructions.

Constraints:
- no extra libraries
- no notebooks
- keep dataset/evaluation logic simple

Success criteria:
- reliable CLI behavior
- run traceability
- clean, beginner-friendly code

## 5) Git Workflow (Branch + PR)

**⚠️ IMPORTANT: Read [GIT_WORKFLOW.md](GIT_WORKFLOW.md) for detailed Git instructions!**

The workshop follows a structured Git workflow:
- Main branch: `main` (protected)
- Development branch: `develop` (base for features)
- Feature branches: `feat/<your-name>-<feature-name>`
- Tags: `v1.0.0`, `v1.1.0`, etc. for versions

Quick start:
```bash
git checkout -b feat/<your-name>-baseline
git add submissions/baseline/model.pkl submissions/baseline/metrics.json submissions/baseline/report.md
git commit -m "Add baseline Titanic model results"
git push -u origin feat/<your-name>-baseline
```

Then open a Pull Request on GitHub to merge your branch into `develop`.

## 6) Rules

- No need to modify files in `src/`.
- No need to change the dataset files in `data/`.
- No need to change the train/test split settings.
- Keep the workflow local: train, save artifacts, commit, and open PR.
- No need for API and no deployment for this workshop.
- **Always work in feature branches, never directly on `main` or `develop`**
- **All PRs must target `develop` branch**
- **Tag releases after merging to `main`**

## 7) Workshop Objective

Learn a practical end-to-end ML collaboration loop:
- load data
- train a simple model
- evaluate with accuracy and F1
- save artifacts
- collaborate through Git branches and Pull Requests

## 📚 Additional Resources

- [Git Workflow Guide](GIT_WORKFLOW.md) - Detailed Git instructions for the workshop
- [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic) - Original competition page
