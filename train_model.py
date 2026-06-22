import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

data = pd.read_csv("dataset/disease.csv")

X = data.drop("Disease", axis=1)
y = data["Disease"]

model = RandomForestClassifier()

model.fit(X, y)

joblib.dump(model, "model/disease_model.pkl")

print("Model Trained Successfully")