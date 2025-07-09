import os
from git import Repo
from dotenv import load_dotenv

load_dotenv()

repo_url = os.getenv('REPO_URL')
clone_dir = os.getenv('CLONE_DIR')

def clone_repo(repo_url, clone_dir):
    if not os.path.exists(clone_dir):
        print(f'Cloning repository from {repo_url}...')
        Repo.clone_from(repo_url, clone_dir)
        print('Repository cloned.')
    else:
        print('Repository already cloned.')

if __name__ == '__main__':
    clone_repo(repo_url, clone_dir)
    print('Ready for data preprocessing.')