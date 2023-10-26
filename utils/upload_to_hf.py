import argparse

from huggingface_hub import HfApi


def main(folder_path: str, repo: str, token: str):
    api = HfApi(token=token)
    print("ensure repo...")
    api.create_repo(
        repo_id=repo, token=token, private=True, repo_type="model", exist_ok=True
    )
    print("start uploading...")
    api.upload_folder(
        folder_path=folder_path,
        repo_id=repo,
        repo_type="model",
    )
    print("uploaded")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Upload model to huggingface')
    parser.add_argument('--folder_path', type=str, required=True ,help='path to folder')
    parser.add_argument('--repo', type=str, required=True ,help='repo name')
    parser.add_argument('--token', type=str, required=True ,help='huggingface token')
    args = parser.parse_args()
    
    main(args.folder_path, args.repo, args.token)
