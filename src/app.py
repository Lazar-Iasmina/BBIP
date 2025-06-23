from flask import Flask, render_template, request, redirect, url_for, jsonify
import pandas as pd
import os

from anomaly_detector import number_of_annomalies
from anomaly_rules import detect_combined_anomalies


app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
csv_path = os.path.join(UPLOAD_FOLDER, "current.csv")

@app.route("/", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files["file"]
        if file and file.filename.endswith(".csv"):
            file.save(csv_path)
            return redirect(url_for("view_data"))
        return "Invalid file format. Please upload a CSV."
    return render_template("upload.html")

@app.route("/view")
def view_data():
    return render_template("view.html")

@app.route("/api/data")
def get_data():
    try:
        df = pd.read_csv(csv_path)

        print(f"✅ Loaded dataset with {len(df)} entries.")

        anomalies_df = detect_combined_anomalies(df)

     # Filter only rows flagged as anomalies
        anomalies_only = anomalies_df[anomalies_df['combined_anomaly'] == True]

        print(f"🔍 Detected {number_of_annomalies} combined anomalies.")

        anomalies_only.to_csv('combined_anomalies.csv', index=False)
        #print("🔍 Results saved to 'combined_anomalies.csv'.")

        df = pd.read_csv('combined_anomalies.csv')
        df = df.replace([float('inf'), float('-inf')], None)  # înlocuiește Infinity cu null
        return jsonify(df.to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
