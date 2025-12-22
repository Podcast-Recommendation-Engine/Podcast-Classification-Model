import mlflow
import pandas as pd

mlflow.set_tracking_uri("http://localhost:5000")

model_name = "podcast-kid-friendly-classifier"

# Load model and get model URI
model_uri = f"models:/{model_name}/Production"
model = mlflow.pyfunc.load_model(model_uri)

# Print model metadata using MlflowClient
client = mlflow.MlflowClient()
model_versions = client.get_latest_versions(model_name, stages=["Production"])
if model_versions:
    model_info = model_versions[0]
    print("Model Metadata:")
    print(f"  Name: {model_info.name}")
    print(f"  Version: {model_info.version}")
    print(f"  Run ID: {model_info.run_id}")
    print(f"  Current Stage: {model_info.current_stage}")
    print(f"  Creation Timestamp: {model_info.creation_timestamp}")


test_texts = [
    "violence "
]

# Convert to DataFrame with the expected column name
test_df = pd.DataFrame({"keywords_text": test_texts})

predictions = model.predict(test_df)

print("\nPredictions:")
for text, pred in zip(test_texts, predictions):
    label = "Kid-Friendly" if pred == 1 else "Not Kid-Friendly"
    print(f"Text: {text}\nPrediction: {label}\n")
