from huggingface_hub import HfApi
api = HfApi()

# Create a new model repository on Hugging Face (e.g. "my-t5-summarizer")
api.create_repo(repo_id="Senjuti3/my-t5-summarizer", repo_type="model")

# Upload your local folder
api.upload_folder(
    folder_path="./saved_summary_model",
    repo_id="Senjuti3/my-t5-summarizer",
    repo_type="model"
)