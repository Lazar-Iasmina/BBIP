import pandas as pd
from anomaly_detector import number_of_annomalies
from anomaly_rules import detect_combined_anomalies

def main():
    csv_path = 'cazuri.csv'  # Replace with your actual file path
    df = pd.read_csv(csv_path)

    print(f"✅ Loaded dataset with {len(df)} entries.")

    anomalies_df = detect_combined_anomalies(df)

    # Filter only rows flagged as anomalies
    anomalies_only = anomalies_df[anomalies_df['combined_anomaly'] == True]

    print(f"🔍 Detected {number_of_annomalies} combined anomalies.")

    anomalies_only.to_csv('combined_anomalies.csv', index=False)
    print("🔍 Results saved to 'combined_anomalies.csv'.")

if __name__ == "__main__":
    main()
