from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os


repo_id = "Bhanu15/Tourism-Package-Prediction-MLOps"
repo_type = "dataset" 

# Initialize an API Client
api = HfApi(token=os.getenv("HF_TOKEN"))

# Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' created.")

api.upload_folder(
    folder_path = "tourism_project/data",
    repo_id = repo_id,
    repo_type = repo_type,
)
