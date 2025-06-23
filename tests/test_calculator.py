import unittest
import sys
import os
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from src.anomaly_rules import *

class TestAnomalyDetection(unittest.TestCase):

    def setUp(self):
        # Example dataframe for testing
        data = {
            'id_cases': [1, 2, 3, 4],
            'age_v': [25, 30, 200, 40],  # 200 is an unrealistic age
            'greutate': [70, 80, 1000, 85],  # 1000 is an unrealistic weight
            'inaltime': [170, 180, 300, 160],  # 300 is an unrealistic height
            'imcINdex': [22, 25, 50, 35],  # BMI values are within normal/abnormal range
            'sex_v': ['male', 'female', 'other', 'female']  # 'other' should be invalid
        }
        self.df = pd.DataFrame(data)

    def test_parse_float_valid(self):
        valid_value = '123.45'
        self.assertEqual(parse_float(valid_value), 123.45)

    def test_parse_float_invalid(self):
        invalid_value = 'invalid'
        self.assertIsNone(parse_float(invalid_value))

    def test_normalize_scores(self):
        scores = [0.2, 0.4, 0.6, 0.8]
        normalized_scores = normalize_scores(scores)
        self.assertTrue(all(1 <= score <= 100 for score in normalized_scores))

    def test_detect_ml_anomalies(self):
        df_ml = self.df.drop(columns=['id_cases', 'sex_v'])  # Drop non-numeric columns for ML testing
        final_ml_anomalies, avg_score, results = detect_ml_anomalies(df_ml)
        self.assertEqual(len(final_ml_anomalies), len(df_ml))  # Check that anomalies are predicted for all rows
        self.assertEqual(len(avg_score), len(df_ml))  # Check that average scores are returned for all rows

    def test_empty_dataframe(self):
        empty_df = pd.DataFrame(columns=['id_cases', 'age_v', 'greutate', 'inaltime', 'imcINdex', 'sex_v'])
        df_with_anomalies = detect_rule_anomalies(empty_df)
        self.assertEqual(len(df_with_anomalies), 0)

    def test_detect_combined_anomalies(self):
        df_combined = detect_combined_anomalies(self.df)
        self.assertEqual(len(df_combined), len(self.df))  # Ensure the output dataframe has the same length
        self.assertIn('combined_anomaly', df_combined.columns)  # Check if 'combined_anomaly' column exists
        self.assertTrue(df_combined['combined_anomaly'].any())  # Ensure that at least one anomaly is detected

    def test_invalid_sex_value(self):
        df_with_anomalies = detect_rule_anomalies(self.df)
        # Row 3 has invalid 'sex_v' value ('other')
        self.assertTrue(df_with_anomalies.loc[2, 'rule_anomaly'])

    def test_unrealistic_age(self):
        df_with_anomalies = detect_rule_anomalies(self.df)
        # Row 3 has unrealistic age (200)
        self.assertTrue(df_with_anomalies.loc[2, 'rule_anomaly'])

    def test_unrealistic_weight(self):
        df_with_anomalies = detect_rule_anomalies(self.df)
        # Row 3 has unrealistic weight (1000)
        self.assertTrue(df_with_anomalies.loc[2, 'rule_anomaly'])

    def test_unrealistic_height(self):
        df_with_anomalies = detect_rule_anomalies(self.df)
        # Row 3 has unrealistic height (300)
        self.assertTrue(df_with_anomalies.loc[2, 'rule_anomaly'])

    def test_unrealistic_bmi(self):
        df_with_anomalies = detect_rule_anomalies(self.df)
        # Row 3 has unrealistic BMI (50 is abnormal but not flagged)
        self.assertTrue(df_with_anomalies.loc[2, 'rule_anomaly'])

    def test_ml_anomalies_with_more_features(self):
        df_ml = self.df.copy()
        df_ml['extra_feature'] = [1, 2, 3, 4]  # Add a non-numeric column for testing
        final_ml_anomalies, avg_score, results = detect_ml_anomalies(df_ml)
        self.assertEqual(len(final_ml_anomalies), len(df_ml))  # Ensure length is correct

    def test_no_anomalies(self):
        # Create a dataframe with no anomalies
        data = {
            'id_cases': [1, 2, 3, 4],
            'age_v': [30, 25, 28, 32],
            'greutate': [70, 80, 75, 85],
            'inaltime': [170, 180, 160, 175],
            'imcINdex': [22, 24, 23, 25],
            'sex_v': ['male', 'female', 'male', 'female']
        }
        df_no_anomalies = pd.DataFrame(data)
        df_with_anomalies = detect_rule_anomalies(df_no_anomalies)
        self.assertFalse(df_with_anomalies['rule_anomaly'].any())  # Ensure no anomalies are detected

    def test_valid_sex_value(self):
        # Testing for valid 'sex_v' values
        data = {
            'id_cases': [1, 2],
            'age_v': [25, 30],
            'greutate': [70, 80],
            'inaltime': [170, 180],
            'imcINdex': [22, 25],
            'sex_v': ['male', 'female']
        }
        df_valid_sex = pd.DataFrame(data)
        df_with_anomalies = detect_rule_anomalies(df_valid_sex)
        self.assertFalse(df_with_anomalies['rule_anomaly'].any())  # Ensure no anomalies for valid sex values

    def test_multiple_invalid_entries(self):
        # Test with multiple invalid entries for sex and age
        data = {
            'id_cases': [1, 2],
            'age_v': [200, 180],  # unrealistic ages
            'greutate': [70, 80],
            'inaltime': [170, 180],
            'imcINdex': [22, 25],
            'sex_v': ['other', 'other']  # invalid sex values
        }
        df_invalid_entries = pd.DataFrame(data)
        df_with_anomalies = detect_rule_anomalies(df_invalid_entries)
        self.assertTrue(df_with_anomalies.loc[0, 'rule_anomaly'])  # Expect anomaly for row 0 (invalid sex)
        self.assertTrue(df_with_anomalies.loc[1, 'rule_anomaly'])  # Expect anomaly for row 1 (invalid sex and age)

    
    def test_unrealistic_height_with_negative_values(self):
        # Test that negative height values are flagged as anomalies
        data = {
            'id_cases': [1, 2],
            'age_v': [25, 30],
            'greutate': [70, 80],
            'inaltime': [-170, 180],  # negative height value
            'imcINdex': [22, 25],
            'sex_v': ['male', 'female']
        }
        df_negative_height = pd.DataFrame(data)
        df_with_anomalies = detect_rule_anomalies(df_negative_height)
        self.assertTrue(df_with_anomalies.loc[0, 'rule_anomaly'])  # Expect anomaly for row 0 (negative height)

    def test_unrealistic_bmi_with_extreme_values(self):
        # Test extreme BMI values that should be flagged as anomalies
        data = {
            'id_cases': [1, 2],
            'age_v': [25, 30],
            'greutate': [70, 80],
            'inaltime': [170, 180],
            'imcINdex': [10, 100],  # extreme BMI values
            'sex_v': ['male', 'female']
        }
        df_extreme_bmi = pd.DataFrame(data)
        df_with_anomalies = detect_rule_anomalies(df_extreme_bmi)
        self.assertTrue(df_with_anomalies.loc[1, 'rule_anomaly'])  # Expect anomaly for row 1 (extremely high BMI)

    def test_invalid_numeric_column(self):
        # Test that invalid numeric columns are handled
        df_invalid_column = self.df.copy()
        df_invalid_column['age_v'] = ['invalid', 30, 40, 50]  # invalid age value
        df_with_anomalies = detect_rule_anomalies(df_invalid_column)
        self.assertTrue(df_with_anomalies.loc[0, 'rule_anomaly'])  # Expect anomaly for row 0 (invalid age)


    def test_detect_ml_anomalies_with_high_dimensional_data(self):
        # Test that high-dimensional data doesn't cause issues
        high_dim_data = self.df.copy()
        for i in range(100):  # Add 100 extra features
            high_dim_data[f'feature_{i}'] = [i for i in range(len(self.df))]
        final_ml_anomalies, avg_score, results = detect_ml_anomalies(high_dim_data)
        self.assertEqual(len(final_ml_anomalies), len(high_dim_data))  # Ensure correct number of anomalies

 

    def test_edge_case_small_dataframe(self):
        # Test that a small dataframe with one row doesn't break the anomaly detection
        data = {
            'id_cases': [1],
            'age_v': [25],
            'greutate': [70],
            'inaltime': [170],
            'imcINdex': [22],
            'sex_v': ['male']
        }
        df_small = pd.DataFrame(data)
        df_with_anomalies = detect_rule_anomalies(df_small)
        self.assertFalse(df_with_anomalies['rule_anomaly'].any())  # No anomalies expected in this case

    def test_invalid_column_data_type(self):
        # Test that invalid data types in columns are flagged as anomalies
        data = {
            'id_cases': [1, 2],
            'age_v': ['invalid', 30],  # Invalid type for age_v (string instead of int)
            'greutate': [70, 80],
            'inaltime': [170, 180],
            'imcINdex': [22, 25],
            'sex_v': ['male', 'female']
        }
        df_invalid_type = pd.DataFrame(data)
        df_with_anomalies = detect_rule_anomalies(df_invalid_type)
        self.assertTrue(df_with_anomalies.loc[0, 'rule_anomaly'])  # Expect anomaly for row 0 (invalid age type)

    
if __name__ == '__main__':
    unittest.main()
