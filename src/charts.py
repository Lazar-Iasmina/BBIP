import pandas as pd
import matplotlib.pyplot as plt

# Load the combined anomalies file
file_path = './combined_anomalies.csv'
df = pd.read_csv(file_path)

# Convert columns to numeric where necessary
df['age_v'] = pd.to_numeric(df['age_v'], errors='coerce')
df['greutate'] = pd.to_numeric(df['greutate'], errors='coerce')
df['inaltime'] = pd.to_numeric(df['inaltime'], errors='coerce')
df['imcINdex'] = pd.to_numeric(df['imcINdex'], errors='coerce')
df['anomaly_score'] = pd.to_numeric(df['anomaly_score'], errors='coerce')
df['data1'] = pd.to_datetime(df['data1'], errors='coerce')

# Drop rows with invalid or missing dates
df = df.dropna(subset=['data1'])

# Plot 1: Anomaly score distribution
plt.figure()
df['anomaly_score'].hist(bins=50)
plt.title('Anomaly Score Distribution')
plt.xlabel('Anomaly Score')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.savefig('./anomaly_score_distribution.png')

# Plot 2: Scatter plot of Weight vs Height colored by Anomaly Score
plt.figure()
plt.scatter(df['inaltime'], df['greutate'], c=df['anomaly_score'], cmap='viridis', alpha=0.7)
plt.colorbar(label='Anomaly Score')
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.title('Height vs Weight colored by Anomaly Score')
plt.grid(True)
plt.tight_layout()
plt.savefig('./height_weight_scatter.png')

# Plot 3: Anomalies over Time
df_time = df[df['combined_anomaly'] == True]
df_time_grouped = df_time.groupby(df_time['data1'].dt.date).size()

plt.figure()
df_time_grouped.plot(kind='line', marker='o')
plt.title('Detected Anomalies Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Anomalies')
plt.grid(True)
plt.tight_layout()
plt.savefig('./anomalies_over_time.png')
