
print("Loading the data")
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data # shape (150, 4)
y = iris.target # shape (150,)
print(iris.feature_names, iris.target_names)

print("Splitting into training and test sets")
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Initializing the model")
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=42)

print("Training (fit) the model")
model.fit(X_train, y_train)

print("Making Predictions")
y_pred = model.predict(X_test)
print("Predictions:", y_pred[:5])
print("True labels:", y_test[:5])

print("Outputing accuracy")
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("Saving the confusion matrix as a png image")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
disp.figure_.savefig('outputs/confusion_matrix.png', dpi=300, bbox_inches='tight')

print("Saving the model as a joblib file")
import joblib
joblib.dump(model, 'outputs/iris_classifier_model.joblib')
