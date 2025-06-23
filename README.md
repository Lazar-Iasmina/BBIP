# Doctor31 Anomaly Detection 🚨

## 📌 Project Overview

This project is part of the **Doctor31 initiative** and focuses on identifying data anomalies within a simulated medical dataset. The goal is to improve **data integrity** and support **clinical decision-making** by flagging suspicious, inconsistent, or biologically implausible records.

It includes both **rule-based detection** and **machine learning-based detection** methods to catch a wide variety of errors.

---

## 📁 Dataset

The dataset is a `.csv` file containing over 7,000 anonymized entries with the following fields:

- `id_cases`: Unique case ID  
- `age_v`: Patient age  
- `sex_v`: Biological sex  
- `agreement`: Legal confirmation status  
- `greutate`: Weight (kg)  
- `inaltime`: Height (cm)  
- `IMC`: BMI category (e.g., Overweight)  
- `data1`: Timestamp  
- `finalizat`: Completion flag  
- `testing`: Testing flag  
- `imcINdex`: BMI numeric value

---

## 🧠 Features

### ✅ Rule-Based Anomaly Detection

- Unrealistic age, height, or weight values  
- Invalid BMI ranges (<12 or >60)  
- Implausible combinations (e.g., obesity in patients >85 years old)  
- Non-standard sex values

### ✅ Machine Learning Anomaly Detection

- `Isolation Forest`
- `One-Class SVM`
- `Local Outlier Factor`
- `Elliptic Envelope`

These are combined to assign an **anomaly score (1–100)** for each record.

---

## 📊 Outputs

- `combined_anomalies.csv`: sorted list of anomalies with explanations and scores
- ML model summary printed to console
- Anomaly score per record (average across ML methods)

---
 python3 -m pytest tests/test_calculator.py


