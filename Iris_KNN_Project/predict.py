"""Load the saved model and classify a new flower.
Usage: python predict.py 5.1 3.5 1.4 0.2
       (sepal_length sepal_width petal_length petal_width, all in cm)
"""
import sys
import joblib
import numpy as np

if len(sys.argv) != 5:
    sys.exit(__doc__)

bundle = joblib.load("models/iris_knn_model.joblib")
x = np.array([[float(v) for v in sys.argv[1:]]])
pred = bundle["model"].predict(bundle["scaler"].transform(x))[0]
print(f"Predicted species: {bundle['classes'][pred]}")
