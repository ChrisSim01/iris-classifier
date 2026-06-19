
# Load the data
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data # shape (150, 4)
y = iris.target # shape (150,)
print(iris.feature_names, iris.target_names)

# Split into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=42)

# Train (fit) the model
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)
print("Predictions:", y_pred[:5])
print("True labels:", y_test[:5])

# Output accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

