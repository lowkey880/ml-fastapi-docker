import joblib
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

iris = load_iris()
X, y = iris.data, iris.target

test_size = 0.2
random_state = 42
n_estimators = 100

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=y)

mlflow.set_experiment("iris-classification")

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("test_size", test_size)
    mlflow.log_param("random_state", random_state)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)

    mlflow.sklearn.log_model(model, name="model", registered_model_name="iris-classifier",
                             serialization_format="cloudpickle")

    joblib.dump(model, "model.joblib")

print("Accuracy:", round(accuracy, 3))
print("F1-score:", round(f1, 3))
