# ⚡ Energy Fraud Detection – Decision Support System

This project presents a **machine learning–based decision support system**
for detecting **non-technical losses (energy fraud)** using historical
electricity consumption data.

The application was developed as a **technical challenge** and demonstrates
how data science models can be transformed into **practical tools** for
utilities, research institutes, and regulatory environments.

---

## 🎯 Project Objective

- Identify anomalous consumption patterns associated with energy fraud
- Support inspection teams by prioritizing high-risk consumers
- Provide explainable, auditable, and scalable fraud risk indicators

> ⚠️ This system is intended as a **decision support tool**, not an automated
fraud accusation mechanism.

---

## 🧠 Methodology

### 1. Data Analysis
- Monthly consumption history (12 months per consumer)
- Exploratory analysis to identify fraud-related patterns

### 2. Feature Engineering
Key indicators derived from consumption behavior:
- Average consumption
- Consumption variability (coefficient of variation)
- Abrupt consumption drops
- Temporal consumption trends

### 3. Modeling
- Supervised machine learning model (Random Forest)
- Probabilistic fraud risk score
- Adjustable risk thresholds

### 4. Explainability
- Rule-based explanations for individual risk classification
- Transparent indicators suitable for regulatory environments

---

## 🖥️ Application Features

- 📂 CSV dataset upload
- 📦 Trained model upload (.pkl)
- 🎛️ Adjustable fraud risk thresholds
- 📊 Interactive dashboards and visualizations
- 🧠 Individual risk explanation
- 📄 PDF inspection reports (summary and individual)
- 📋 Inspection priority list

---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/your-username/energy-fraud-detection.git
cd energy-fraud-detection
