# Raja Health Prediction System

An enterprise-grade, single-page application and API replacing the Raja Credit System. This system leverages Machine Learning to predict the likelihood of Heart Disease, Diabetes, and Breast Cancer using premium UI/UX design.

## Features
- **3-in-1 Prediction Engine**: Heart Disease, Diabetes, Breast Cancer.
- **Auto Model Selection**: Trains LR, RF, SVM, XGBoost and auto-selects the best algorithm per disease based on F1 Score.
- **Premium UI**: Single HTML file with React, Tailwind CSS (Glassmorphism), and Chart.js.
- **PDF Reports**: One-click downloadable patient reports.
- **History Tracking**: LocalStorage-based prediction history.
- **Analytics Dashboard**: Visual model performance metrics.

## Folder Structure
```text
raja-health-prediction/
├── backend/
│   ├── app.py              # Flask REST API
│   ├── train.py            # Model training pipeline
│   ├── requirements.txt    # Python dependencies
│   ├── runtime.txt         # Python version for Render
│   └── render.yaml         # Render deployment spec
├── frontend/
│   └── index.html          # Complete React SPA (Zero build tools)
├── datasets/
│   ├── heart.csv
│   ├── diabetes.csv
│   └── breast-cancer.csv
├── models/                 # Auto-generated .pkl and .json files
├── .gitignore
└── README.md