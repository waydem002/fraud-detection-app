# Fraud Detection System

A full-stack machine learning project focused on detecting fraudulent financial transactions using supervised learning and anomaly detection techniques.
This project is designed as a **production-style ML system**, progressing from data exploration to model deployment.

---

## 📌 Problem Statement

Fraudulent transactions represent a very small fraction of total financial activity but cause significant financial loss.
The main challenge is **extreme class imbalance**, where fraud cases account for less than 1% of all transactions, making traditional accuracy-based models ineffective.

This project aims to build a robust fraud detection system that prioritizes:

* Recall (catching fraud)
* Precision (reducing false alarms)
* Real-world deployment readiness

---

## 📊 Dataset

* **Source:** Kaggle – Credit Card Fraud Detection Dataset
* **Transactions:** ~284,000
* **Fraud Rate:** ~0.17%
* **Features:** PCA-transformed numerical features (`V1–V28`)
* **Target Variable:** `Class`

  * `0` → Normal transaction
  * `1` → Fraudulent transaction

The raw dataset is stored untouched to preserve data integrity.

---

## 📁 Project Structure

```
fraud-detection-system/
│
├── data/
│   ├── raw/                # Original datasets (never modified)
│   └── processed/          # Cleaned and feature-engineered data
│
├── notebooks/              # Exploratory analysis and experiments
│
├── src/                    # Reusable production-style code
│   ├── data/               # Data loading and preprocessing
│   ├── features/           # Feature engineering
│   ├── models/             # Model training
│   └── evaluation/         # Metrics and model comparison
│
├── models/                 # Saved trained models
├── reports/                # Final report, figures, screenshots
├── api/                    # Backend API (planned)
├── dashboard/              # Frontend dashboard (planned)
└── README.md
```

---

## ✅ Progress So Far (Week 1)

### ✔ Data Setup & Exploration

* Initialized a production-ready project structure
* Loaded and validated the dataset
* Verified extreme class imbalance
* Explored transaction `Amount` and `Time` features
* Visualized fraud vs non-fraud distributions
* Confirmed absence of missing values

### ✔ Engineering Foundations

* Created reusable data-loading functions inside `src/`
* Separated exploratory notebooks from production code
* Addressed Python import paths for scalable development

### ✔ Key Insights

* Fraud cases represent less than 1% of all transactions
* Transaction amounts are heavily right-skewed
* Fraud behavior differs significantly from normal transactions
* Accuracy alone is not an appropriate evaluation metric

---

## 🛠 Tech Stack

* **Language:** Python
* **Data Analysis:** pandas, numpy
* **Visualization:** matplotlib, seaborn
* **Machine Learning:** scikit-learn, imbalanced-learn
* **Version Control:** Git & GitHub

---

## 🗺 Roadmap

This project follows a structured **24-week roadmap**, progressing through:

* Data preprocessing & feature engineering
* Model development & evaluation
* API and dashboard creation
* Deployment and final presentation

🚧 **Current Status:** Week 1 — Data Exploration & Foundations
