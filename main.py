import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import os
from flask import Flask, request, render_template, redirect, url_for, jsonify

app = Flask(__name__, template_folder='templates')

file_path = "Simulated_Banking_Server_Data.csv"
df_original = pd.read_csv(file_path)
df = df_original.copy()

scalar = StandardScaler()
features1 = ['CPU_Usage(%)', 'Memory_Usage(%)', 'Disk_IO(MB/s)', 'Network_Latency(ms)', 'Error_Rate']
features2 = ['CPU_Usage(%)', 'Memory_Usage(%)', 'Disk_IO(MB/s)', 'Network_Latency(ms)', 'Error_Rate', 'Downtime']
df[features1] = scalar.fit_transform(df[features1])

df0 = df[df['Downtime'] == 0]
df1 = df[df['Downtime'] == 1]

model = IsolationForest(contamination = 0.05)
model.fit(df[features1])

def anomaly_detection (anomaly_score):
    if anomaly_score > 0.06:
        # print("Not an anomaly")
        return 0
    else:
        # print("Anomaly")
        return 1

def model_prediction (CPU_Usage, Memory_Usage, Disk_IO, Network_Latency, Error_Rate):
    values = np.array([[CPU_Usage, Memory_Usage, Disk_IO, Network_Latency, Error_Rate]])
    values = pd.DataFrame(values)
    # scalar.fit(training_set)
    values = scalar.transform(values)
    # last_value = np.array([[Downtime]])
    # values = np.append(values, last_value, axis=1)
    anomaly_score = model.decision_function(values)
    # print(anomaly_score)
    return anomaly_detection(anomaly_score)

# scalar.fit(df_original[features1])
# model_prediction(55, 50, 163.931450, 30, 0.058)

@app.route("/", methods=["GET", "POST"])
def predict_anomaly():
    prediction = None
    CPU_Usage = Memory_Usage = Disk_IO = Network_Latency = Error_Rate = None

    if request.method == "POST":
        CPU_Usage = request.form.get("CPU_Usage")
        Memory_Usage = request.form.get("Memory_Usage")
        Disk_IO = request.form.get("Disk_IO")
        Network_Latency = request.form.get("Network_Latency")
        Error_Rate = request.form.get("Error_Rate")

        # Get inputs from form and parse them to floats
        CPU_Usage = float(CPU_Usage) if CPU_Usage else None
        Memory_Usage = float(Memory_Usage) if Memory_Usage else None
        Disk_IO = float(Disk_IO) if Disk_IO else None
        Network_Latency = float(Network_Latency) if Network_Latency else None
        Error_Rate = float(Error_Rate) if Error_Rate else None

        # Call the model prediction function
        if None not in (CPU_Usage, Memory_Usage, Disk_IO, Network_Latency, Error_Rate):
            prediction = model_prediction(CPU_Usage, Memory_Usage, Disk_IO, Network_Latency, Error_Rate)

    # Render the HTML template with prediction
    return render_template("index.html", prediction=prediction, CPU_Usage=CPU_Usage, Memory_Usage=Memory_Usage, Disk_IO=Disk_IO, Network_Latency=Network_Latency, Error_Rate=Error_Rate)

if __name__ == '__main__':
    app.run()


