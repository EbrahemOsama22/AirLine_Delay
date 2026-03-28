
## Airline Delay Cause Analysis ✈️⏱️

A full end-to-end data science project that analyzes airline delay patterns across years and builds deep learning models to both classify the main cause of delays and predict the delay rate.

📎 **Dataset:** [Kaggle — Airline Delay Cause](https://www.kaggle.com/datasets/abdelazizel7or/airline-delay-cause)

---

### 📌 What This Project Does

- **Data Cleaning** — Removes missing values, drops redundant identifier columns, and renames columns for clarity and consistency.
- **Feature Engineering** — Derives new meaningful features including `problem_rate`, `on_time_rate`, `delay_rate`, `season`, and per-cause delay ratios (`carrier_delay_ratio`, `weather_delay_ratio`, `nas_delay_ratio`, etc.).
- **Exploratory Data Analysis (EDA)** — Visualizes:
  - Average total flights and delay rates per year
  - Seasonal delay patterns by month
  - Top 10 airlines and airports by delay rate
  - Delay cause breakdown per carrier and per airport
  - Year-over-year trends for each delay cause

---

### 🤖 Machine Learning Models

Both models are built using **TensorFlow / Keras** with the **AdamW optimizer** and **Early Stopping**.

#### 1️⃣ Delay Cause Classification
A **multi-class classification model** that predicts the main cause of a flight delay — `Carrier`, `Weather`, `NAS`, `Security`, or `Late Aircraft` — using flight statistics and engineered features.

| Detail | Value |
|---|---|
| Architecture | Dense Neural Network (128 → 64 → 32 → Softmax) |
| Loss Function | Categorical Cross-Entropy |
| Handling Imbalance | Class Weights (balanced) |
| Evaluation | Accuracy, Classification Report, Confusion Matrix |

---

#### 2️⃣ Delay Rate Regression
A **regression model** that predicts the `delay_rate` (proportion of flights delayed more than 15 minutes) for a given carrier/airport/time combination.

| Detail | Value |
|---|---|
| Architecture | Dense Neural Network (128 → 64 → Dropout(0.3) → 32 → Linear) |
| Loss Function | Mean Squared Error (MSE) |
| Metrics | MAE, MSE |
| Evaluation | Loss & MAE curves over epochs |

---

### 🛠️ Tech Stack

`Python` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `Scikit-learn` · `TensorFlow` · `Keras`

---

