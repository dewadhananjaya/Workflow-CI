import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import os

def load_data(file_path):
    df = pd.read_csv(file_path)
    X = df.drop("Class", axis=1)
    y = df["Class"]
    return X, y

def train_and_log_model(X_train, X_test, y_train, y_test, n_neighbors=5):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Gunakan run aktif jika ada, kalau tidak buat run baru
    active_run = mlflow.active_run()
    if active_run is None:
        run_context = mlflow.start_run(run_name="KNN-DryBean")
    else:
        run_context = mlflow.start_run(run_name="KNN-DryBean", nested=True)

    with run_context:
        mlflow.sklearn.autolog(log_models=True)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print(f"✅ Model dilatih, akurasi: {acc:.4f}")

        mlflow.log_metric("accuracy", acc)

        report = classification_report(y_test, y_pred, output_dict=True)
        report_path = "classification_report.csv"
        pd.DataFrame(report).transpose().to_csv(report_path)
        mlflow.log_artifact(report_path)

        mlflow.sklearn.log_model(model, "model")

        print("📦 Model dan metrik berhasil dicatat di MLflow")

def main():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment("Dry_Bean_Classification_KNN_Autolog")

    X, y = load_data("Preprocessed_Dry_Bean.csv")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_and_log_model(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()