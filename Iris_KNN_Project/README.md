# Data Classification Using AI: Iris Species Classifier (KNN)

**DecodeLabs | Artificial Intelligence Industrial Training Kit | Project 2 | Batch 2026**

A supervised-learning project that trains a **K-Nearest Neighbors (KNN)** model to classify iris flowers into three species (*setosa*, *versicolor*, *virginica*) from four measurements, and validates it with a confusion matrix, precision, recall and F1 score.

---

## 1. Project Summary

| Item | Detail |
|---|---|
| **Goal** | Build a basic classification model using a small dataset |
| **Dataset** | Iris benchmark: 150 samples, 3 balanced classes, 4 features |
| **Features** | Sepal length, sepal width, petal length, petal width (cm) |
| **Algorithm** | K-Nearest Neighbors (the "proximity principle") |
| **Split** | 80% training / 20% testing, shuffled and stratified |
| **Preprocessing** | `StandardScaler` (mean = 0, variance = 1) |
| **Validation** | Confusion matrix, precision, recall, F1 score |
| **Key skills** | Data handling, supervised learning basics, model training |

The project follows the kit's **IPO framework**:

```
INPUT                    PROCESS                       OUTPUT
Iris dataset      ->     Train/Test split      ->      Confusion matrix
Feature scaling          KNN algorithm (tuned K)       Precision / Recall / F1
```

### Results (test set of 30 unseen flowers)

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---|---|---|
| Baseline KNN (K = 5, from the kit) | 93.3% | 0.944 | 0.933 | 0.933 |
| **Final KNN (K = 12, tuned by cross-validation)** | **96.7%** | **0.970** | **0.967** | **0.967** |

Final confusion matrix (rows = actual, columns = predicted):

| | setosa | versicolor | virginica |
|---|---|---|---|
| **setosa** | 10 | 0 | 0 |
| **versicolor** | 0 | 10 | 0 |
| **virginica** | 0 | 1 | 9 |

*Setosa is separated perfectly. The only error is one virginica predicted as versicolor, which is expected because those two species overlap.*

> Note: with only 30 test samples, one flower changes accuracy by 3.3 points. K = 5, 10 and 12 tie for the lowest cross-validation error (3.33%); the script breaks ties toward the larger, smoother K. Results are reproducible via `random_state=42`.

---

## 2. Step-by-Step Process

1. **Set up the environment.** Install Python 3.9+ and the packages in `requirements.txt`.
2. **Load and understand the data.** Load Iris, check shape, missing values, class balance and feature ranges; plot the classes.
3. **Split before anything else.** Shuffle (removes order bias, since the raw data is sorted by species) and split 80/20 with stratification. The test set stays "locked" until final evaluation.
4. **Scale the features.** Apply `StandardScaler`. **Fit on the training set only**, then transform both sets, so no test information leaks into training.
5. **Choose K.** Test K = 1..30 with 5-fold cross-validation on the training data and pick the "elbow" (lowest error). K = 1 risks overfitting; very large K underfits.
6. **Train the model.** The scikit-learn workflow: *instantiate* `KNeighborsClassifier`, *fit* on `X_train`, *predict* on `X_test`.
7. **Validate the output.** Build the confusion matrix and compute precision, recall and F1 (accuracy alone can mislead).
8. **Test on brand-new data.** Predict species for flowers the model has never seen (`predict.py`).
9. **Experiment (bonus).** Compare KNN with Logistic Regression, Decision Tree and SVM.
10. **Save and document.** Save the model with joblib, store plots and metrics, write this README.

---

## 3. Project Structure

```
iris_knn_project/
├── iris_classification.py      # Main end-to-end pipeline
├── predict.py                  # Classify a new flower from the command line
├── requirements.txt            # Dependencies
├── README.md                   # This file
├── models/
│   └── iris_knn_model.joblib   # Trained model + scaler + class names
└── outputs/
    ├── 01_data_exploration.png # Class scatter plots
    ├── 02_k_tuning_curve.png   # Error rate vs K (elbow plot)
    ├── 03_confusion_matrix.png # Final confusion matrix
    ├── algorithm_comparison.csv# KNN vs other algorithms
    ├── results.json            # All metrics
    └── run_log.txt             # Full console output of the run
```

---

## 4. How to Run

```bash
# 1. (optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt

# 3. run the full pipeline
python iris_classification.py

# 4. classify a new flower (sepal_length sepal_width petal_length petal_width)
python predict.py 6.9 3.1 5.8 2.1
# -> Predicted species: virginica
```

---

## 5. Key Concepts Applied

- **Supervised learning:** we do not write the rules; we provide labelled history and the machine derives the logic.
- **Feature scaling:** KNN uses distances, so features with larger ranges would dominate unless standardised.
- **Train/test split with shuffle:** the test set measures performance on data the model has not seen.
- **KNN:** a new point takes the majority class of its K closest training points.
- **Choosing K:** small K = noisy/overfit, large K = generic/underfit; the elbow is the sweet spot.
- **Confusion matrix:** TP, FP (false alarm, Type I), FN (missed detection, Type II), TN.
- **Precision vs recall vs F1:** precision = trustworthiness of positive predictions, recall = how many real positives are found, F1 = their harmonic mean.

## 6. Bonus: Algorithm Comparison (5-fold CV, macro F1, training data)

| Algorithm | Mean F1 |
|---|---|
| KNN (K = 12) | 0.966 |
| SVM (RBF) | 0.966 |
| Logistic Regression | 0.958 |
| Decision Tree | 0.941 |

## 7. Future Work

Move from tabular data to image data with **Deep Learning and CNNs** (Computer Vision), as previewed in the kit's "Emerging Horizons" slide.

---

*Built for the DecodeLabs AI Internship, Project 2.*
