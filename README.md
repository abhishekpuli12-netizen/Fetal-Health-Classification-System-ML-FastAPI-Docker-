# 🩺 Fetal Health Classification System (ML + FastAPI + Docker)

## 📌 Project Overview

This project predicts fetal health condition using cardiotocography (CTG) data.
It classifies fetal health into:

* 🟢 **Normal (1)**
* 🟡 **Suspect (2)**
* 🔴 **Pathological (3)**

---

## 🚀 What Makes This Project Strong

* ✅ End-to-end ML pipeline (data → model → API)
* ✅ XGBoost-based classification model
* ✅ Explainable AI using **SHAP**
* ✅ REST API using FastAPI
* ✅ Dockerized deployment

---

## 🏗️ Project Structure

```text
fetal-health-classification/
│
├── app/
│   ├── main.py              # FastAPI application (API endpoints)
│   └── schema.py            # Request validation (Pydantic)
│
├── src/
│   ├── data_loader.py       # Load dataset
│   ├── preprocess.py        # Data preprocessing
│   ├── train.py             # Model training logic
│   └── predict.py           # Model loading & prediction
│
├── artifacts/
│   └── model.pkl            # Trained model
│
│__fetal_health.csv     # Dataset
│
├── notebooks/
│   └── fetal_project.ipynb  # EDA + SHAP analysis
│
│__plots/
|    |___ROC-curve.png
|    |___Waterfall_plot
|    |___SHAP Summary Bar Plot
|
|__ train_pipeline.py    # Training pipeline   
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

* Source: Kaggle (Fetal Health Dataset)
* Type: Structured tabular data
* Features: 21 numerical features derived from CTG signals
* Target: `fetal_health` (multi-class classification)

---

## ⚙️ Machine Learning Pipeline

### 1️⃣ Data Ingestion

* Loaded dataset using Pandas
* Checked missing values and consistency

---

### 2️⃣ Preprocessing

* Feature-target split
* Label transformation `[1,2,3] → [0,1,2]`
* Train-test split (80/20)

👉 No feature scaling used because:

> Tree-based models like XGBoost do not require normalization

---

### 3️⃣ Model Training

```python
XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6
)
```

* Class imbalance handled using **sample weights**
* Model saved using **joblib**

---

## 📈 Model Performance (My Model)

```
              precision    recall  f1-score   support

         0.0       0.98      0.96      0.97       333
         1.0       0.84      0.91      0.87        64
         2.0       0.91      1.00      0.95        29

    accuracy                           0.96       426
```

### 🔍 Key Observations

* Strong performance on **Normal & Pathological**
* 🔥 High recall for **Suspect class (0.91)**
* ✅ Perfect detection of **Pathological cases (1.00 recall)**

---

## 📊 Comparison with Kaggle Baseline

### 🔹 Kaggle Model

```
              precision    recall  f1-score   support

         1.0       0.98      0.98      0.98       333
         2.0       0.89      0.89      0.89        64
         3.0       1.00      0.97      0.98        29

    accuracy                           0.96
```

---

### 🔹 Improvement

| Metric                | Kaggle | My Model    |
| --------------------- | ------ | ----------- |
| Accuracy              | ~0.96  | ~0.96       |
| Recall (Suspect)      | 0.89   | **0.91 ⬆️** |
| Recall (Pathological) | 0.97   | **1.00 ⬆️** |

---

### 🧠 Insight

> Improving recall is critical in healthcare, where missing risky cases is more dangerous than false alarms.

---

## 🔍 Model Explainability (SHAP)

SHAP (SHapley Additive exPlanations) was used to interpret model predictions.

### 📈 SHAP Summary Plot

---

### 🔍 Interpretation

* 🔥 **abnormal_short_term_variability** is the most influential feature
* 📊 **percentage_of_time_with_abnormal_long_term_variability** also plays a major role
* ⚡ **accelerations** significantly impact predictions

---

### 🧠 Insight

The model relies heavily on **variability-related features**, which aligns with medical understanding of fetal monitoring.

---

## 🌐 API Deployment (FastAPI)

### Endpoint

```
POST /predict
```

---

### Sample Request

```json
{
  "baseline_value": 120,
  "accelerations": 0.005,
  "fetal_movement": 0.02,
  "uterine_contractions": 0.002,
  "light_decelerations": 0.0,
  "severe_decelerations": 0.0,
  "prolongued_decelerations": 0.0,
  "abnormal_short_term_variability": 20,
  "mean_value_of_short_term_variability": 1.5,
  "percentage_of_time_with_abnormal_long_term_variability": 10,
  "mean_value_of_long_term_variability": 10,
  "histogram_width": 60,
  "histogram_min": 90,
  "histogram_max": 150,
  "histogram_number_of_peaks": 3,
  "histogram_number_of_zeroes": 0,
  "histogram_mode": 120,
  "histogram_mean": 130,
  "histogram_median": 125,
  "histogram_variance": 50,
  "histogram_tendency": 1
}
```

---

### Sample Response

```json
{
  "prediction": 1,
  "label": "Normal",
  "probabilities": [0.88, 0.11, 0.01]
}
```

---

## 🐳 Docker Deployment

### Build Image

```bash
docker build -t fetal-api .
```

---

### Run Container

```bash
docker run -p 8000:8000 fetal-api
```

---

### Access API

```
http://localhost:8000/docs
```

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* FastAPI
* Docker
* SHAP

---

## 🧠 Key Learnings

* Accuracy alone is not sufficient in medical ML
* Recall is critical for detecting risky cases
* Tree-based models do not require feature scaling
* Explainability is essential in healthcare AI
* Deployment is crucial for real-world ML systems

---

## 📌 Future Improvements

* 🔹 Integrate SHAP into API
* 🔹 Cloud deployment (AWS / Render)
* 🔹 Frontend UI (Streamlit)
* 🔹 Uncertainty-aware predictions

---

## ⚠️ Disclaimer

This project is for educational purposes only and should not be used for real medical decisions without clinical validation.

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub!
