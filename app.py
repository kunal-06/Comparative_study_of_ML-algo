import os
import pandas as pd
from flask import Flask, request, render_template, jsonify
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"

# Supported algorithms
ALGORITHMS = {
    "Logistic Regression": LogisticRegression,
    "Random Forest": RandomForestClassifier,
    "SVM": SVC,
    "KNN": KNeighborsClassifier,
    "Decision Tree": DecisionTreeClassifier,
    "Naive Bayes": GaussianNB,
    "Gradient Boosting": GradientBoostingClassifier,
}


def preprocess_dataset(dataset_path):
    """Preprocess the dataset dynamically."""
    df = pd.read_csv(dataset_path)

    # Handle duplicates
    df = df.drop_duplicates()

    # Handle missing values
    for col in df.columns:
        if df[col].dtype == "object":  # Categorical column
            df[col] = df[col].fillna(df[col].mode()[0])  # Fill with mode
        else:  # Numeric column
            df[col] = df[col].fillna(df[col].median())  # Fill with median

    # Separate features (X) and target (y)
    X = df.iloc[:, :-1]  # Features: all columns except last
    y = df.iloc[:, -1]   # Target: last column

    # Convert categorical features to numerical
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = LabelEncoder().fit_transform(X[col])

    # Convert target column to numerical if categorical
    if y.dtype == "object":
        y = LabelEncoder().fit_transform(y)

    # Scale numeric features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the dataset into train and test sets
    return train_test_split(X, y, test_size=0.2, random_state=42)



@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Handle dataset upload
        file = request.files.get("dataset")
        if file and file.filename.endswith(".csv"):
            dataset_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(dataset_path)

            try:
                # Preprocess dataset
                X_train, X_test, y_train, y_test = preprocess_dataset(dataset_path)

                # Train and test all algorithms
                results = {}
                for name, AlgoClass in ALGORITHMS.items():
                    model = AlgoClass()
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)

                    #if predictions[0] in y_test:
                    accuracy = accuracy_score(y_test, predictions)
                    #else:
                        #accuracy = mean_absolute_error(y_train, predictions)
                    print(accuracy)
                    results[name] = accuracy

                return jsonify(results)

            except Exception as e:
                # Handle errors during preprocessing or training
                return jsonify({"error": f"An error occurred: {str(e)}"})

    return render_template("index.html")


if __name__ == "__main__":
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    app.run(debug=True)

                   
