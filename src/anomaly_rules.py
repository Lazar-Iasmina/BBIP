import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import MinMaxScaler


def parse_float(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def normalize_scores(scores):
    scaler = MinMaxScaler(feature_range=(1, 100))
    scores = np.array(scores).reshape(-1, 1)
    return scaler.fit_transform(scores).flatten()


def detect_rule_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    anomalies = []
    df = df.copy()
    df['rule_anomaly'] = False
    for idx, row in df.iterrows():
        anomaly_reasons = []

        age = parse_float(row['age_v'])
        weight = parse_float(row['greutate'])
        height_cm = parse_float(row['inaltime'])
        bmi = parse_float(row['imcINdex'])
        sex = str(row['sex_v']).lower() if pd.notna(row['sex_v']) else None

        if age is None or age <= 0 or age > 120:
            anomaly_reasons.append(f"Unrealistic age: {row['age_v']}")
        if weight is None or weight < 20 or weight > 300:
            anomaly_reasons.append(f"Unrealistic weight: {row['greutate']}")
        if height_cm is None or height_cm < 120 or height_cm > 220:
            anomaly_reasons.append(f"Unrealistic height: {row['inaltime']}")
        if bmi is None or bmi < 12 or bmi > 60:
            anomaly_reasons.append(f"BMI out of normal range: {row['imcINdex']}")
        if age is not None and bmi is not None and age > 85 and bmi >= 30:
            anomaly_reasons.append(f"Suspicious elderly obesity: age {age}, BMI {bmi}")
        if sex not in ('male', 'female'):
            anomaly_reasons.append(f"Invalid sex value: {row['sex_v']}")

        if anomaly_reasons:
            df.at[idx, 'rule_anomaly'] = True
            anomalies.append({
                'id_cases': row['id_cases'],
                'anomalies': '; '.join(anomaly_reasons)
            })

    return df


def detect_ml_anomalies(df: pd.DataFrame):
    numeric_features = df.select_dtypes(include=['number']).copy()

    # Clean up the numeric data
    numeric_features.replace([np.inf, -np.inf], np.nan, inplace=True)
    numeric_features.fillna(numeric_features.median(numeric_only=True), inplace=True)
    numeric_features = numeric_features.clip(lower=-1e6, upper=1e6)

    results = {}
    anomaly_matrix = []

    # Isolation Forest
    iso = IsolationForest(contamination=0.05, random_state=42)
    iso_preds = iso.fit_predict(numeric_features)
    iso_scores = normalize_scores(-iso.decision_function(numeric_features))
    results['isolation_forest'] = (iso_preds == -1, iso_scores)
    anomaly_matrix.append(iso_preds == -1)

    # One-Class SVM
    svm = OneClassSVM(nu=0.05, kernel="rbf", gamma='scale')
    svm_preds = svm.fit_predict(numeric_features)
    svm_scores = normalize_scores(-svm.decision_function(numeric_features))
    results['one_class_svm'] = (svm_preds == -1, svm_scores)
    anomaly_matrix.append(svm_preds == -1)

    # Local Outlier Factor
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
    lof_preds = lof.fit_predict(numeric_features)
    lof_scores = normalize_scores(-lof.negative_outlier_factor_)
    results['local_outlier_factor'] = (lof_preds == -1, lof_scores)
    anomaly_matrix.append(lof_preds == -1)

    # Elliptic Envelope
    try:
        ee = EllipticEnvelope(contamination=0.05, random_state=42)
        ee_preds = ee.fit_predict(numeric_features)
        ee_scores = normalize_scores(-ee.mahalanobis(numeric_features))
        results['elliptic_envelope'] = (ee_preds == -1, ee_scores)
        anomaly_matrix.append(ee_preds == -1)
    except:
        # If it fails, treat as no anomalies
        zero_preds = np.zeros(len(df), dtype=bool)
        ones_score = np.ones(len(df))
        results['elliptic_envelope'] = (zero_preds, ones_score)
        anomaly_matrix.append(zero_preds)

    # Final ML decision: anomaly if any model flagged it
    final_ml_anomalies = np.any(np.column_stack(anomaly_matrix), axis=1)

    # Average the scores
    all_scores = np.column_stack([v[1] for v in results.values()])
    avg_score = np.mean(all_scores, axis=1)

    return final_ml_anomalies, avg_score, results


def detect_combined_anomalies(df: pd.DataFrame):
    df = df.copy()
    df = detect_rule_anomalies(df)

    ml_preds, anomaly_scores, model_details = detect_ml_anomalies(df)

    df['ml_anomaly'] = ml_preds
    df['anomaly_score'] = anomaly_scores
    df['combined_anomaly'] = df['rule_anomaly'] | df['ml_anomaly']

    # Sort by anomaly score
    df = df.sort_values(by='anomaly_score', ascending=False)

    # Print summary
    print("\n ML Anomaly Model Summary:")
    for name, (preds, _) in model_details.items():
        print(f"- {name}: {np.sum(preds)} anomalies")

    #print(f"\n Total combined anomalies: {df['combined_anomaly'].sum()}")
    return df
