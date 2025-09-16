# for data manipulation
import pandas as pd
import os
# for data preprocessing and splitting
from sklearn.model_selection import train_test_split
# for hugging face space authentication to upload files
from huggingface_hub import HfApi

# Initialize Hugging Face API with token from environment variable
api = HfApi(token=os.getenv("HF_TOKEN"))

# Path to dataset in Hugging Face
DATASET_PATH = "hf://datasets/Bhanu15/Tourism-Package-Prediction-MLOps/tourism.csv"

# Load dataset
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully with shape:", df.shape)

# Drop unnecessary columns
drop_cols = []
for col in ["Unnamed: 0", "CustomerID"]:
    if col in df.columns:
        drop_cols.append(col)

if drop_cols:
    df.drop(columns=drop_cols, inplace=True)
    print(f"Dropped columns: {drop_cols}")

# Define target variable
target = "ProdTaken"

# Define predictors (all columns except target)
X = df.drop(columns=[target])
y = df[target]

# Split dataset into training and test sets (80/20)
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save splits locally
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Train/Test splits saved locally.")

# Files to upload
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

# Upload files back to Hugging Face dataset repo
for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # save with same filename
        repo_id="Bhanu15/Tourism-Package-Prediction-MLOps",
        repo_type="dataset",
    )
    print(f"Uploaded {file_path} to Hugging Face dataset repo.")
