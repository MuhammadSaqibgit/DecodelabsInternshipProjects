"""
DecodeLabs | Artificial Intelligence | Project 2
Data Classification Using AI  (Iris dataset + K-Nearest Neighbors)

Pipeline (IPO framework from the project kit)
    INPUT   : Load Iris dataset -> explore -> feature scaling
    PROCESS : Shuffled train/test split (80/20) -> KNN training -> tune K
    OUTPUT  : Confusion matrix, precision / recall / F1, saved model

Run:  python iris_classification.py
"""

import json
import os
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")  # save plots to files; works without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TEST_SIZE = 0.20          # 80% train / 20% test
BASELINE_K = 5            # value used in the project kit
K_RANGE = range(1, 31)    # K values searched during tuning
OUT_DIR = "outputs"
MODEL_DIR = "models"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def banner(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


# --------------------------------------------------------------------------
# STEP 1 - INPUT: load and understand the dataset
# --------------------------------------------------------------------------
def load_and_explore():
    banner("STEP 1: LOAD AND UNDERSTAND THE DATASET")
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df["species"] = df["target"].map(dict(enumerate(iris.target_names)))

    print(f"Samples   : {df.shape[0]}")
    print(f"Features  : {len(iris.feature_names)} -> {iris.feature_names}")
    print(f"Classes   : {len(iris.target_names)} -> {[str(c) for c in iris.target_names]}")
    print(f"Missing   : {int(df.isna().sum().sum())} values")
    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))
    print("\nClass balance:")
    print(df["species"].value_counts().to_string())
    print("\nStatistical summary (note the different feature ranges):")
    print(df[iris.feature_names].describe().round(2).to_string())

    # Plot 1: petal length vs petal width, coloured by species
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = ["#1f3b73", "#e8590c", "#2f9e44"]
    for idx, name in enumerate(iris.target_names):
        sub = df[df["species"] == name]
        axes[0].scatter(sub["petal length (cm)"], sub["petal width (cm)"],
                        label=name, color=colors[idx], alpha=0.8, edgecolor="k")
        axes[1].scatter(sub["sepal length (cm)"], sub["sepal width (cm)"],
                        label=name, color=colors[idx], alpha=0.8, edgecolor="k")
    axes[0].set(xlabel="Petal length (cm)", ylabel="Petal width (cm)",
                title="Petal features separate the classes well")
    axes[1].set(xlabel="Sepal length (cm)", ylabel="Sepal width (cm)",
                title="Sepal features overlap more")
    axes[0].legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/01_data_exploration.png", dpi=150)
    plt.close()

    return iris, df


# --------------------------------------------------------------------------
# STEP 2 - PROCESS: shuffled train/test split, then scaling
# --------------------------------------------------------------------------
def split_and_scale(iris, df):
    banner("STEP 2: TRAIN / TEST SPLIT (80/20, SHUFFLED) + FEATURE SCALING")
    X = df[iris.feature_names].values
    y = df["target"].values

    # shuffle=True randomises order (removes order bias: the raw file is sorted
    # by species). stratify=y keeps the 3 classes balanced in both sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        shuffle=True,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    print(f"Training set : {X_train.shape[0]} samples")
    print(f"Testing set  : {X_test.shape[0]} samples (locked until final evaluation)")

    # Gatekeeper rule: StandardScaler (mean=0, variance=1).
    # IMPORTANT: fit on TRAIN only, then transform both -> no data leakage.
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    print("\nAfter scaling (train set):")
    print("  mean     :", np.round(X_train_s.mean(axis=0), 3))
    print("  variance :", np.round(X_train_s.var(axis=0), 3))

    return X_train, X_test, X_train_s, X_test_s, y_train, y_test, scaler


# --------------------------------------------------------------------------
# STEP 3 - PROCESS: tune K (elbow method) using cross-validation on TRAIN only
# --------------------------------------------------------------------------
def tune_k(X_train, y_train):
    banner("STEP 3: TUNING THE ENGINE - CHOOSING K")
    errors = []
    for k in K_RANGE:
        # scaler inside the pipeline is re-fitted per fold -> leakage-free CV
        pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
        acc = cross_val_score(pipe, X_train, y_train, cv=5, scoring="accuracy").mean()
        errors.append(1 - acc)

    errors = np.array(errors)
    # Smallest error wins; ties are broken toward the LARGER (smoother) K.
    best_idx = int(np.where(errors == errors.min())[0][-1])
    best_k = list(K_RANGE)[best_idx]

    print(f"Lowest 5-fold CV error rate : {errors.min():.4f}  at K = {best_k}")
    print(f"Error rate at K=1 (overfit risk): {errors[0]:.4f}")
    print(f"Error rate at K=30             : {errors[-1]:.4f}")

    plt.figure(figsize=(8, 4.5))
    plt.plot(list(K_RANGE), errors, marker="o", color="#1f3b73")
    plt.scatter([best_k], [errors[best_idx]], s=300, facecolors="none",
                edgecolors="#e8590c", linewidths=3, label=f"Optimal K = {best_k}")
    plt.xlabel("K value (number of neighbours)")
    plt.ylabel("Error rate (5-fold CV)")
    plt.title("Tuning the engine: choosing K")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/02_k_tuning_curve.png", dpi=150)
    plt.close()
    return best_k


# --------------------------------------------------------------------------
# STEP 4 - PROCESS: train the KNN model (instantiate -> fit -> predict)
# --------------------------------------------------------------------------
def train_and_predict(k, X_train_s, X_test_s, y_train):
    model = KNeighborsClassifier(n_neighbors=k)   # INSTANTIATE (build the frame)
    model.fit(X_train_s, y_train)                 # FIT        (memorise the map)
    preds = model.predict(X_test_s)               # PREDICT    (apply logic)
    return model, preds


# --------------------------------------------------------------------------
# STEP 5 - OUTPUT: validation (confusion matrix, precision, recall, F1)
# --------------------------------------------------------------------------
def evaluate(name, y_test, preds, class_names, save_prefix=None):
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="macro")
    rec = recall_score(y_test, preds, average="macro")
    f1 = f1_score(y_test, preds, average="macro")
    cm = confusion_matrix(y_test, preds)

    print(f"\n--- {name} ---")
    print(f"Accuracy        : {acc:.4f}")
    print(f"Precision (macro): {prec:.4f}")
    print(f"Recall (macro)   : {rec:.4f}")
    print(f"F1 score (macro) : {f1:.4f}")
    print("\nConfusion matrix (rows = actual, columns = predicted):")
    print(pd.DataFrame(cm, index=class_names, columns=class_names).to_string())
    print("\nClassification report:")
    print(classification_report(y_test, preds, target_names=class_names, digits=3))

    if save_prefix:
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
            ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"Confusion matrix - {name}")
        plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/{save_prefix}.png", dpi=150)
        plt.close()

    return {"accuracy": acc, "precision_macro": prec,
            "recall_macro": rec, "f1_macro": f1, "confusion_matrix": cm.tolist()}


# --------------------------------------------------------------------------
# BONUS - compare algorithms (suggested in the project conclusion)
# --------------------------------------------------------------------------
def compare_algorithms(X_train, y_train, best_k):
    banner("BONUS: COMPARING DIFFERENT ALGORITHMS (5-fold CV on training data)")
    candidates = {
        f"KNN (K={best_k})": KNeighborsClassifier(n_neighbors=best_k),
        "Logistic Regression": LogisticRegression(max_iter=500),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "SVM (RBF)": SVC(),
    }
    rows = []
    for name, clf in candidates.items():
        pipe = make_pipeline(StandardScaler(), clf)
        scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="f1_macro")
        rows.append({"Algorithm": name, "Mean F1": scores.mean(), "Std": scores.std()})
        print(f"{name:<22} F1 = {scores.mean():.4f} (+/- {scores.std():.4f})")
    pd.DataFrame(rows).round(4).to_csv(f"{OUT_DIR}/algorithm_comparison.csv", index=False)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    iris, df = load_and_explore()
    X_train, X_test, X_train_s, X_test_s, y_train, y_test, scaler = split_and_scale(iris, df)
    class_names = [str(c) for c in iris.target_names]

    best_k = tune_k(X_train, y_train)

    banner("STEP 4 & 5: TRAIN MODELS AND VALIDATE ON THE LOCKED TEST SET")
    base_model, base_preds = train_and_predict(BASELINE_K, X_train_s, X_test_s, y_train)
    base_metrics = evaluate(f"Baseline KNN (K={BASELINE_K})", y_test, base_preds, class_names)

    final_model, final_preds = train_and_predict(best_k, X_train_s, X_test_s, y_train)
    final_metrics = evaluate(f"Final KNN (K={best_k})", y_test, final_preds, class_names,
                             save_prefix="03_confusion_matrix")

    compare_algorithms(X_train, y_train, best_k)

    # Test the model on completely new, unseen flowers
    banner("PREDICTING NEW, UNSEEN FLOWERS")
    new_flowers = np.array([
        [5.0, 3.4, 1.5, 0.2],   # looks like setosa
        [6.0, 2.9, 4.5, 1.5],   # looks like versicolor
        [6.9, 3.1, 5.8, 2.1],   # looks like virginica
    ])
    new_pred = final_model.predict(scaler.transform(new_flowers))
    for row, p in zip(new_flowers, new_pred):
        print(f"{row.tolist()}  ->  {class_names[p]}")

    # Save artefacts
    joblib.dump({"model": final_model, "scaler": scaler, "classes": class_names},
                f"{MODEL_DIR}/iris_knn_model.joblib")
    results = {"best_k": best_k,
               "baseline_k5": base_metrics,
               "final_model": final_metrics,
               "train_size": int(len(y_train)), "test_size": int(len(y_test))}
    with open(f"{OUT_DIR}/results.json", "w") as f:
        json.dump(results, f, indent=2)

    banner("DONE - results saved in 'outputs/' and model saved in 'models/'")


if __name__ == "__main__":
    main()
