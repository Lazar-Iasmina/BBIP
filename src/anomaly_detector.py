import pandas as pd
number_of_annomalies=2086
class AnomalyDetector:
    
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.anomalies = pd.DataFrame()

    def calculate_bmi(self, weight, height_cm):
        height_m = height_cm / 100
        return weight / (height_m ** 2)

    def detect(self):
        self.df['BMI'] = self.calculate_bmi(self.df['greutate'], self.df['inaltime'])

        conditions = {
            'age_invalid': self.df['age_v'] > 120,
            'weight_invalid': self.df['greutate'] < 30,
            'bmi_too_low': self.df['BMI'] < 12,
            'bmi_too_high': self.df['BMI'] > 60,
        }

        for label, condition in conditions.items():
            self.df[label] = condition


        anomaly_mask = self.df[list(conditions.keys())].any(axis=1)
        self.anomalies = self.df[anomaly_mask]

        return self.anomalies

    def explain(self, row):
        reasons = []
        if row['age_invalid']: reasons.append("Age > 120")
        if row['weight_invalid']: reasons.append("Weight < 30kg")
        if row['bmi_too_low']: reasons.append("BMI < 12")
        if row['bmi_too_high']: reasons.append("BMI > 60")
        return reasons