import mlflow
import pandas as pd

mlflow.set_tracking_uri("http://localhost:5000")

model_name = "podcast-kid-friendly-classifier"

model = mlflow.pyfunc.load_model(f"models:/{model_name}/Production")

test_texts = [
    "kids children education learning",
    "violence crime murder adult"
]

# Convert to DataFrame with the expected column name
test_df = pd.DataFrame({"keywords_text": test_texts})

predictions = model.predict(test_df)

for text, pred in zip(test_texts, predictions):
    label = "Kid-Friendly" if pred == 1 else "Not Kid-Friendly"
    print(f"Text: {text}\nPrediction: {label}\n")
