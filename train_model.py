import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')   # non-interactive backend, saves to file
import matplotlib.pyplot as plt
import seaborn as sns

with open("data/dataset.pkl", "rb") as f:
    dataset = pickle.load(f)

X = dataset["data"]    # shape (2600, 42)
y = dataset["labels"]  # list of 2600 letters

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Labels: {sorted(set(y))}")


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2, # 80% train, 20% test

    random_state=42, # Makes the split reproducible, same result every time you run it

    stratify=y # Ensures each letter is proportionally represented in both splits
   
)

print(f"Train: {len(X_train)} | Test: {len(X_test)}")

#  Train Random Forest 
print("Training model...")
model = RandomForestClassifier(
    n_estimators=200, # Build 200 decision trees.

    max_depth=None, # Let each tree grow until all leaves are pure (contain only one class).

    min_samples_split=2, # A node splits if it has at least 2 samples.

    random_state=42, # Makes training reproducible.

    n_jobs=-1 # Use all available CPU cores in parallel.
)

model.fit(X_train, y_train) # the magic command, the model looks at the coordinates and leanrs patterns.
y_pred = model.predict(X_test) # the model is given coordinates from the 20% test, and the ans is stored in y_pred

acc = accuracy_score(y_test, y_pred) # % of letters it got right
print(f"\nAccuracy: {acc * 100:.2f}%") 

print("\nPer-letter breakdown:")
print(classification_report(y_test, y_pred)) # depper breakdown, tells where the model is struggling letter wise


# Confusion matrix 
# A 26x26 grid. Row = true label, Column = predicted label.
# Diagonal = correct. Off-diagonal = mistakes.
cm = confusion_matrix(y_test, y_pred, labels=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))

plt.figure(figsize=(16, 14))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    xticklabels=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
    yticklabels=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
    cmap='Blues'
)
plt.title(f'Confusion Matrix | Accuracy: {acc*100:.1f}%')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
print("Confusion matrix saved to confusion_matrix.png")

# save the model, this is the trained brain
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model saved to model.pkl")