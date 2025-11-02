import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import mlflow
import mlflow.sklearn

# =========================
# 1️⃣ Konfigurasi MLflow
# =========================
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(tracking_uri)
print(f"🔗 MLflow Tracking URI: {tracking_uri}")

mlflow.set_experiment("dry-bean-experiment")
mlflow.autolog(log_models=True)

# =========================
# 2️⃣ Load Dataset
# =========================
def load_data(file_path="Preprocessed_Dry_Bean.csv"):
    df = pd.read_csv(file_path)
    X = df.drop(['Class'], axis=1)
    y = df['Class']
    return X, y

# =========================
# 3️⃣ Training Function
# =========================
def train_model(data_path):
    X, y = load_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = KNeighborsClassifier(n_neighbors=5)

    # Jalankan training dan logging tanpa start_run()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    print(f"✅ Model dilatih, akurasi: {acc:.4f}")
    mlflow.log_metric("accuracy", acc)

    # Simpan classification report ke artifact
    report_path = "classification_report.csv"
    pd.DataFrame(report).transpose().to_csv(report_path)
    mlflow.log_artifact(report_path)

    # Simpan model manual (backup)
    mlflow.sklearn.log_model(model, "model")

    print("📦 Model dan metric telah dicatat di MLflow")

# =========================
# 4️⃣ Main
# =========================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="Preprocessed_Dry_Bean.csv")
    args = parser.parse_args()

    train_model(args.data_path)