import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt


def run_models(X_train, y_train, X_test, y_test, class_names=None):
    # Scale (important for LR & SVM)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    results = {}

    # -------------------------------------------------------
    # Baseline — Majority Class
    # -------------------------------------------------------
    most_common_class = Counter(y_train).most_common(1)[0][0]
    y_pred_baseline = np.full_like(y_test, most_common_class)

    results["Baseline"] = {
        "accuracy": accuracy_score(y_test, y_pred_baseline),
        "f1": f1_score(y_test, y_pred_baseline, average="macro")
    }

    # -------------------------------------------------------
    # Logistic Regression 
    # -------------------------------------------------------
    print("Training LR")
    logreg = LogisticRegression(max_iter=1000, C=0.01)
    logreg.fit(X_train, y_train)

    y_pred_logreg = logreg.predict(X_test)
    results["Logistic Regression"] = {
        "accuracy": accuracy_score(y_test, y_pred_logreg),
        "f1": f1_score(y_test, y_pred_logreg, average="macro")
    }
    print("LR Done")

    # -------------------------------------------------------
    # PCA + SVM
    # -------------------------------------------------------
    print("Training SVM")

    pca = PCA(n_components=50)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    svm = LinearSVC(C=1, max_iter=5000)
    svm.fit(X_train_pca, y_train)
    y_pred_svm = svm.predict(X_test_pca)

    results["PCA + Linear SVM"] = {
        "accuracy": accuracy_score(y_test, y_pred_svm),
        "f1": f1_score(y_test, y_pred_svm, average="macro")
    }

    print("SVM Done")

    # -------------------------------------------------------
    # Decision Tree
    # -------------------------------------------------------
    
    print("Training DT")

    dt = DecisionTreeClassifier(max_depth=5, min_samples_leaf=25, random_state=42)
    dt.fit(X_train, y_train)

    y_pred_dt = dt.predict(X_test)
    results["Decision Tree"] = {
        "accuracy": accuracy_score(y_test, y_pred_dt),
        "f1": f1_score(y_test, y_pred_dt, average="macro")
    }

    print("DT Done")


    # -------------------------------------------------------
    # Accuracy + F1 Table
    # -------------------------------------------------------
    print("\n=== Performance Comparison (Accuracy & F1) ===")
    for model, scores in results.items():
        print(f"{model:25} {scores['accuracy']:.4f}     {scores['f1']:.4f}")

    # -------------------------------------------------------
    # Confusion Matrix (best model)
    # -------------------------------------------------------
    best_model = max(results, key=lambda k: results[k]["accuracy"])

    if best_model == "Baseline":
        y_pred_best = y_pred_baseline
    elif best_model == "Logistic Regression":
        y_pred_best = y_pred_logreg
    elif best_model == "Decision Tree":
        y_pred_best = y_pred_dt
    elif best_model == "PCA + Linear SVM":
        y_pred_best = y_pred_svm

    cm = confusion_matrix(y_test, y_pred_best)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(f"Confusion Matrix — {best_model}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    return results
