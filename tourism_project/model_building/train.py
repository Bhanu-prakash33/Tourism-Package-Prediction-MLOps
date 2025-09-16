# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow

# ----------------------------
# MLflow setup
# ----------------------------
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlops-training-experiment-tourism")

api = HfApi()

# ----------------------------
# Load train/test splits from Hugging Face dataset repo
# ----------------------------
Xtrain_path = "hf://datasets/Bhanu15/Tourism-Package-Prediction-MLOps/Xtrain.csv"
Xtest_path = "hf://datasets/Bhanu15/Tourism-Package-Prediction-MLOps/Xtest.csv"
ytrain_path = "hf://datasets/Bhanu15/Tourism-Package-Prediction-MLOps/ytrain.csv"
ytest_path = "hf://datasets/Bhanu15/Tourism-Package-Prediction-MLOps/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)

# If y trains/ tests are single-column dataframes, squeeze to Series
if isinstance(ytrain, pd.DataFrame) and ytrain.shape[1] == 1:
    ytrain = ytrain.squeeze()
if isinstance(ytest, pd.DataFrame) and ytest.shape[1] == 1:
    ytest = ytest.squeeze()

print("Loaded Xtrain, Xtest, ytrain, ytest shapes:")
print(Xtrain.shape, Xtest.shape, ytrain.shape, ytest.shape)

# ----------------------------
# Feature lists
# ----------------------------
numeric_features = [
    "Age",
    "CityTier",
    "DurationOfPitch",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "PreferredPropertyStar",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "MonthlyIncome",
]

categorical_features = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "ProductPitched",
    "MaritalStatus",
    "Designation",
]

# Verify features exist in Xtrain
numeric_features = [c for c in numeric_features if c in Xtrain.columns]
categorical_features = [c for c in categorical_features if c in Xtrain.columns]

# ----------------------------
# Handle class imbalance via scale_pos_weight for xgboost
# ----------------------------
# compute ratio negative/positive
value_counts = ytrain.value_counts()
if 1 in value_counts and 0 in value_counts:
    scale_pos_weight = value_counts[0] / value_counts[1]
else:
    scale_pos_weight = 1.0

print("scale_pos_weight:", scale_pos_weight)

# ----------------------------
# Preprocessing and pipeline
# ----------------------------
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
)

xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss",
                              scale_pos_weight=scale_pos_weight, random_state=42)

model_pipeline = make_pipeline(preprocessor, xgb_model)

# ----------------------------
# Hyperparameter grid
# ----------------------------
param_grid = {
    "xgbclassifier__n_estimators": [50, 100],
    "xgbclassifier__max_depth": [3, 5],
    "xgbclassifier__learning_rate": [0.01, 0.1],
    "xgbclassifier__colsample_bytree": [0.5, 0.8],
    "xgbclassifier__reg_lambda": [0.1, 1.0],
}

# ----------------------------
# Grid search with cross-validation
# ----------------------------
grid_search = GridSearchCV(model_pipeline, param_grid, cv=4, n_jobs=-1, verbose=1)

# Start MLflow run
with mlflow.start_run():
    print("Starting GridSearchCV...")
    grid_search.fit(Xtrain, ytrain)

    # Log cv results: each param set as nested run (as in sample)
    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        param_set = results["params"][i]
        mean_score = results["mean_test_score"][i]
        std_score = results["std_test_score"][i]

        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", float(mean_score))
            mlflow.log_metric("std_test_score", float(std_score))

    # Log best parameters in parent run
    best_params = grid_search.best_params_
    mlflow.log_params(best_params)
    print("Best params:", best_params)

    # Retrieve best estimator
    best_model = grid_search.best_estimator_

    # Use a classification threshold 
    classification_threshold = 0.5

    # Predict probabilities and binary predictions
    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    # Classification reports
    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Log metrics
    mlflow.log_metrics({
        "train_accuracy": float(train_report["accuracy"]),
        "train_precision": float(train_report.get("1", {}).get("precision", 0.0)),
        "train_recall": float(train_report.get("1", {}).get("recall", 0.0)),
        "train_f1_score": float(train_report.get("1", {}).get("f1-score", 0.0)),
        "test_accuracy": float(test_report["accuracy"]),
        "test_precision": float(test_report.get("1", {}).get("precision", 0.0)),
        "test_recall": float(test_report.get("1", {}).get("recall", 0.0)),
        "test_f1_score": float(test_report.get("1", {}).get("f1-score", 0.0)),
    })

    # Save best model locally
    model_filename = "best_tourism_xgb_model_v1.joblib"
    joblib.dump(best_model, model_filename)
    print(f"Saved best model to {model_filename}")

    # Log model artifact
    mlflow.log_artifact(model_filename, artifact_path="model")

    # ----------------------------
    # Upload model to Hugging Face model hub
    # ----------------------------
    repo_id = "Bhanu15/Tourism-Package-Prediction-MLOps"
    repo_type = "model"

    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Model repo '{repo_id}' exists. Proceeding to upload.")
    except RepositoryNotFoundError:
        print(f"Model repo '{repo_id}' not found. Creating...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Model repo '{repo_id}' created.")

    # Upload model artifact file
    try:
        api.upload_file(
            path_or_fileobj=model_filename,
            path_in_repo=model_filename,
            repo_id=repo_id,
            repo_type=repo_type,
        )
        print(f"Uploaded model file '{model_filename}' to Hugging Face repo '{repo_id}'.")
    except HfHubHTTPError as e:
        print("Failed to upload to Hugging Face:", e)
