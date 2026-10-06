"""Root entry point so `gcloud functions deploy --source .` finds the function."""
from cloud_function.main import analyze, lambda_handler  # noqa: F401
