import os
import pandas as pd
import numpy as np
import time
import traceback
from flask import Flask, request, render_template, jsonify, make_response
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render assigns a port
    app.run(host="0.0.0.0", port=port)


app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["TIMEOUT"] = 300  # 5 minutes timeout for grid search

# Ensure upload folder exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Supported algorithms with their respective parameter grids
CLASSIFICATION_ALGORITHMS = {
    "Logistic Regression": {
        "algorithm": LogisticRegression,
        "params": {
            "C": [0.1, 1.0, 10.0],
            "solver": ["liblinear", "lbfgs"],
            "max_iter": [100, 200]
        }
    },
    "Random Forest": {
        "algorithm": RandomForestClassifier,
        "params": {
            "n_estimators": [50, 100],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5]
        }
    },
    "SVM": {
        "algorithm": SVC,
        "params": {
            "C": [0.1, 1.0, 10.0],
            "kernel": ["linear", "rbf"],
            "gamma": ["scale", "auto"]
        }
    },
    "KNN": {
        "algorithm": KNeighborsClassifier,
        "params": {
            "n_neighbors": [3, 5, 7],
            "weights": ["uniform", "distance"]
        }
    },
    "Decision Tree": {
        "algorithm": DecisionTreeClassifier,
        "params": {
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
            "criterion": ["gini", "entropy"]
        }
    },
    "Naive Bayes": {
        "algorithm": GaussianNB,
        "params": {
            "var_smoothing": [1e-9, 1e-8, 1e-7]
        }
    },
    "Gradient Boosting": {
        "algorithm": GradientBoostingClassifier,
        "params": {
            "n_estimators": [50, 100],
            "learning_rate": [0.01, 0.1],
            "max_depth": [3, 5]
        }
    }
}

REGRESSION_ALGORITHMS = {
    "Linear Regression": {
        "algorithm": LinearRegression,
        "params": {
            "fit_intercept": [True, False],
            "normalize": [True, False]
        }
    },
    "Random Forest": {
        "algorithm": RandomForestRegressor,
        "params": {
            "n_estimators": [50, 100],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5]
        }
    },
    "SVR": {
        "algorithm": SVR,
        "params": {
            "C": [0.1, 1.0, 10.0],
            "kernel": ["linear", "rbf"],
            "gamma": ["scale", "auto"]
        }
    },
    "KNN": {
        "algorithm": KNeighborsRegressor,
        "params": {
            "n_neighbors": [3, 5, 7],
            "weights": ["uniform", "distance"]
        }
    },
    "Decision Tree": {
        "algorithm": DecisionTreeRegressor,
        "params": {
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
            "criterion": ["mse", "mae"]
        }
    },
    "Ridge": {
        "algorithm": Ridge,
        "params": {
            "alpha": [0.1, 1.0, 10.0],
            "solver": ["auto", "svd", "cholesky"]
        }
    },
    "Lasso": {
        "algorithm": Lasso,
        "params": {
            "alpha": [0.1, 1.0, 10.0],
            "max_iter": [1000, 2000]
        }
    },
    "Gradient Boosting": {
        "algorithm": GradientBoostingRegressor,
        "params": {
            "n_estimators": [50, 100],
            "learning_rate": [0.01, 0.1],
            "max_depth": [3, 5]
        }
    }
}


def is_classification(y):
    """Determine if this is a classification or regression problem"""
    # If target is object type or has fewer than 10 unique values, treat as classification
    if y.dtype == "object" or len(np.unique(y)) < 10:
        return True
    return False


def preprocess_dataset(dataset_path):
    """Preprocess the dataset dynamically."""
    print(f"Reading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"Dataset shape: {df.shape}")

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

    # Check if it's a classification or regression problem
    classification_problem = is_classification(y)
    print(f"Problem type detected: {'Classification' if classification_problem else 'Regression'}")

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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, classification_problem


# Simple test route to verify API communication
@app.route("/test", methods=["GET", "POST"])
def test_route():
    if request.method == "POST":
        # Simple echo test
        data = {"message": "API working correctly", "received": "POST request"}
        if 'file' in request.files:
            data['file_received'] = request.files['file'].filename
        return jsonify(data)
    return jsonify({"message": "API working correctly", "received": "GET request"})


@app.route("/", methods=["GET", "POST"])
def index():
    print(f"Request method: {request.method}")
    
    if request.method == "POST":
        print("Files in request:", list(request.files.keys()))
        # Handle dataset upload
        file = request.files.get("dataset")
        
        if not file:
            print("No file in request")
            response = make_response(jsonify({"error": "No file uploaded"}), 400)
            response.headers['Content-Type'] = 'application/json'
            print(response)
            return response
            
        print(f"File received: {file.filename}")
            
        if not file.filename.endswith(".csv"):
            print(f"Invalid file type: {file.filename}")
            response = make_response(jsonify({"error": "Uploaded file must be a CSV file"}), 400)
            response.headers['Content-Type'] = 'application/json'
            return response
            
        try:
            # Save the file
            dataset_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(dataset_path)
            print(f"File saved to: {dataset_path}")
            
            # Check if file exists and is readable
            if not os.path.exists(dataset_path):
                raise Exception(f"File not saved properly at {dataset_path}")
                
            # Get file size
            file_size = os.path.getsize(dataset_path)
            print(f"File size: {file_size} bytes")
            
            if file_size == 0:
                raise Exception("Uploaded file is empty")

            # Preprocess dataset
            X_train, X_test, y_train, y_test, classification_problem = preprocess_dataset(dataset_path)

            # Choose appropriate algorithms based on problem type
            algorithms = CLASSIFICATION_ALGORITHMS if classification_problem else REGRESSION_ALGORITHMS
            
            # Train and test all algorithms with hyperparameter tuning
            results = {}
            best_params = {}
            
            start_time = time.time()
            timeout = app.config["TIMEOUT"]
            
            for name, algo_config in algorithms.items():
                # Check if we're approaching timeout
                if time.time() - start_time > timeout * 0.8:  # Use 80% of timeout as safety margin
                    print(f"Skipping {name} due to approaching timeout")
                    continue
                    
                print(f"Tuning {name}...")
                
                # Create GridSearchCV object with reduced parameter sets if needed
                param_grid = algo_config["params"]
                
                # Create GridSearchCV object
                grid_search = GridSearchCV(
                    algo_config["algorithm"](),
                    param_grid,
                    cv=3,  # 3-fold cross-validation
                    scoring='accuracy' if classification_problem else 'r2',
                    n_jobs=-1 if name != "Naive Bayes" else 1,  # Use all cores except for Naive Bayes
                    verbose=1  # Show progress
                )
                
                try:
                    # Fit the grid search with timeout protection
                    grid_search.fit(X_train, y_train)
                    
                    # Get best model and parameters
                    best_model = grid_search.best_estimator_
                    best_params[name] = grid_search.best_params_
                    
                    # Make predictions using the best model
                    predictions = best_model.predict(X_test)
                    
                    if classification_problem:
                        # For classification problems use accuracy
                        score = accuracy_score(y_test, predictions)
                        metric_name = "Accuracy"
                    else:
                        # For regression problems use R² score (higher is better like accuracy)
                        score = r2_score(y_test, predictions)
                        metric_name = "R² Score"
                    
                    results[name] = score
                    
                except Exception as e:
                    print(f"Error with {name}: {str(e)}")
                    continue

            # Check if we have any results
            if not results:
                response = make_response(jsonify({"error": "No algorithms could be trained successfully"}), 500)
                response.headers['Content-Type'] = 'application/json'
                return response

            # Send problem type to frontend
            response_data = {
                "results": results,
                "problem_type": "Classification" if classification_problem else "Regression",
                "metric": metric_name,
                "best_params": best_params
            }
            
            print("Sending response data:", response_data)
            
            # Create response with explicit Content-Type header
            response = make_response(jsonify(response_data))
            response.headers['Content-Type'] = 'application/json'
            return response

        except Exception as e:
            print(f"Exception in training: {str(e)}")
            print(traceback.format_exc())
            # Handle errors during preprocessing or training
            response = make_response(jsonify({"error": f"An error occurred: {str(e)}"}), 500)
            response.headers['Content-Type'] = 'application/json'
            return response

    return render_template("index.html")


if __name__ == "__main__":
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    app.run(debug=True, threaded=True)

                   
