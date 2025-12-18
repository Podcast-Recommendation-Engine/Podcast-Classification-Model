import mlflow

# Set tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Load production model
model = mlflow.sklearn.load_model("models:/podcast-kid-friendly-classifier/Production")

# Test with sample text
test_texts = [
    "kids children education learning",
    "violence crime murder adult"
]

predictions = model.predict(test_texts)

for text, pred in zip(test_texts, predictions):
    label = "Kid-Friendly" if pred == 1 else "Not Kid-Friendly"
    print(f"Text: {text}\nPrediction: {label}\n")
