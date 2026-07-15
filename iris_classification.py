# Iris Flower Classification Project
# CodeAlpha Internship

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Load Iris Dataset
iris = load_iris()

# Convert dataset into table format
data = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add species column
data["species"] = iris.target

# Display first few rows
print("First 5 Rows:")
print(data.head())

# Input features
X = data.drop("species", axis=1)

# Output
y = data["species"]

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = RandomForestClassifier()

# Train model
model.fit(X_train, y_train)

# Predict species
predictions = model.predict(X_test)

# Check accuracy
accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:")
print(accuracy)

# Detailed report
print("\nClassification Report:")
print(classification_report(y_test, predictions))
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Create confusion matrix
cm = confusion_matrix(y_test, predictions)

# Draw graph
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()