import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt


def run_models(X, y, class_names=None):

    # Flatten if images (N, H, W)
    if len(X.shape) > 2:
        X = X.reshape(X.shape[0], -1)

    print("Data shapes:", X.shape, y.shape)

    # -------------------------------------------------------
    # Train/Test Split
    # -------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale (important for LR & SVM)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # -------------------------------------------------------
    # Baseline — Majority Class
    # -------------------------------------------------------
    most_common_class = Counter(y_train).most_common(1)[0][0]
    y_pred_baseline = [most_common_class] * len(y_test)
    baseline_acc = accuracy_score(y_test, y_pred_baseline)
    baseline_f1 = f1_score(y_test, y_pred_baseline, average='macro')

    print("\nBaseline Model:")
    print("Most common emotion:", most_common_class)
    print("Baseline accuracy:", baseline_acc)

    # -------------------------------------------------------
    # Logistic Regression (Hyperparameter Tuning)
    # -------------------------------------------------------
    print("\nTuning Logistic Regression...")
    logreg_grid = GridSearchCV(
        LogisticRegression(max_iter=1000),
        {"C": [0.1, 1, 5]},
        cv=3,
        scoring='accuracy',
        n_jobs=-1
    )
    logreg_grid.fit(X_train, y_train)
    best_logreg = logreg_grid.best_estimator_

    y_pred_logreg = best_logreg.predict(X_test)
    logreg_acc = accuracy_score(y_test, y_pred_logreg)
    logreg_f1 = f1_score(y_test, y_pred_logreg, average='macro')

    print("Best LR params:", logreg_grid.best_params_)
    print("Logistic Regression accuracy:", logreg_acc)

    # -------------------------------------------------------
    # PCA + SVM (Hyperparameter Tuning)
    # -------------------------------------------------------
    print("\nRunning PCA (100 components)...")
    pca = PCA(n_components=100)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    print("\nTuning Linear SVM...")
    svm_grid = GridSearchCV(
        LinearSVC(),
        {"C": [0.1, 1, 5]},
        cv=3,
        scoring='accuracy',
        n_jobs=-1
    )
    svm_grid.fit(X_train_pca, y_train)
    best_svm = svm_grid.best_estimator_

    y_pred_svm = best_svm.predict(X_test_pca)
    svm_acc = accuracy_score(y_test, y_pred_svm)
    svm_f1 = f1_score(y_test, y_pred_svm, average='macro')

    print("Best SVM params:", svm_grid.best_params_)
    print("Linear SVM (with PCA) accuracy:", svm_acc)

    # -------------------------------------------------------
    # Decision Tree (Hyperparameter Tuning)
    # -------------------------------------------------------
    print("\nTuning Decision Tree...")
    dt_grid = GridSearchCV(
        DecisionTreeClassifier(),
        {
            "max_depth": [5, 10, 20, None],
            "min_samples_split": [2, 5, 10]
        },
        cv=3,
        scoring='accuracy'
    )
    dt_grid.fit(X_train, y_train)
    best_dt = dt_grid.best_estimator_

    y_pred_dt = best_dt.predict(X_test)
    dt_acc = accuracy_score(y_test, y_pred_dt)
    dt_f1 = f1_score(y_test, y_pred_dt, average='macro')

    print("Best DT params:", dt_grid.best_params_)
    print("Decision Tree accuracy:", dt_acc)

    # -------------------------------------------------------
    # Accuracy + F1 Table
    # -------------------------------------------------------
    print("\n=== Performance Comparison (Accuracy & F1) ===")
    print(f"{'Model':25} {'Accuracy':10} {'F1-Score'}")
    print(f"{'Baseline':25} {baseline_acc:.4f}     {baseline_f1:.4f}")
    print(f"{'Logistic Regression':25} {logreg_acc:.4f}     {logreg_f1:.4f}")
    print(f"{'Linear SVM + PCA':25} {svm_acc:.4f}     {svm_f1:.4f}")
    print(f"{'Decision Tree':25} {dt_acc:.4f}     {dt_f1:.4f}")

    # -------------------------------------------------------
    # Confusion Matrix (best model)
    # -------------------------------------------------------
    model_scores = {
        "baseline": baseline_acc,
        "logreg": logreg_acc,
        "svm": svm_acc,
        "dt": dt_acc
    }
    best_model = max(model_scores, key=model_scores.get)
    print("\nBest model:", best_model)

    if best_model == "baseline":
        y_pred_best = y_pred_baseline
    elif best_model == "logreg":
        y_pred_best = y_pred_logreg
    elif best_model == "svm":
        y_pred_best = y_pred_svm
    else:
        y_pred_best = y_pred_dt

    cm = confusion_matrix(y_test, y_pred_best)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(f"Confusion Matrix — {best_model.upper()}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    return model_scores
