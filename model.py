import numpy as np
from resize import X, y   # Load resized images and labels
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC       # <-- FAST LINEAR SVM
from sklearn.tree import DecisionTreeClassifier
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------------------------------
# 1. Train/Test Split
# -------------------------------------------------------
print("Data shapes:", X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------------------------------
# 2. Baseline Model — Majority Class Classifier
# -------------------------------------------------------
most_common_class = Counter(y_train).most_common(1)[0][0]
y_pred_baseline = [most_common_class] * len(y_test)
baseline_acc = accuracy_score(y_test, y_pred_baseline)

print("\nBaseline Model:")
print("Most common emotion:", most_common_class)
print("Baseline accuracy:", baseline_acc)

# -------------------------------------------------------
# 3. Logistic Regression
# -------------------------------------------------------
print("\nTraining Logistic Regression...")
logreg = LogisticRegression(max_iter=300, n_jobs=-1)
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)
logreg_acc = accuracy_score(y_test, y_pred_logreg)
print("Logistic Regression accuracy:", logreg_acc)

# -------------------------------------------------------
# 4. FAST Linear SVM
# -------------------------------------------------------
from sklearn.decomposition import PCA

print("\nRunning PCA dimensionality reduction...")
pca = PCA(n_components=100)     # reduce 3072 -> 100
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print("\nTraining Linear SVM on PCA data...")
svm = LinearSVC()
svm.fit(X_train_pca, y_train)
y_pred_svm = svm.predict(X_test_pca)
svm_acc = accuracy_score(y_test, y_pred_svm)
print("Linear SVM (with PCA) accuracy:", svm_acc)
# -------------------------------------------------------
# 5. Decision Tree
# -------------------------------------------------------
print("\nTraining Decision Tree...")
dt = DecisionTreeClassifier(max_depth=None)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
dt_acc = accuracy_score(y_test, y_pred_dt)
print("Decision Tree accuracy:", dt_acc)

# -------------------------------------------------------
# 6. Accuracy Comparison Table
# -------------------------------------------------------
print("\n=== Accuracy Comparison ===")
print("Baseline:           ", baseline_acc)
print("Logistic Regression:", logreg_acc)
print("Linear SVM:         ", svm_acc)
print("Decision Tree:      ", dt_acc)

# -------------------------------------------------------
# 7. Confusion Matrix for Best Model
# -------------------------------------------------------
accuracies = {
    "baseline": baseline_acc,
    "logreg": logreg_acc,
    "svm": svm_acc,
    "dt": dt_acc
}

best_model_name = max(accuracies, key=accuracies.get)
print("\nBest model:", best_model_name)

if best_model_name == "logreg":
    y_pred_best = y_pred_logreg
elif best_model_name == "svm":
    y_pred_best = y_pred_svm
elif best_model_name == "dt":
    y_pred_best = y_pred_dt
else:
    y_pred_best = y_pred_baseline

cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=False, cmap='Blues')
plt.title(f"Confusion Matrix — {best_model_name.upper()}")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
