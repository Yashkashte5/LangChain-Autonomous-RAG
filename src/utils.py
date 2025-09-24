import os
from pathlib import Path
from dotenv import load_dotenv


def ensure_dirs():
    Path('data/raw').mkdir(parents=True, exist_ok=True)
    Path('data/vectorstore').mkdir(parents=True, exist_ok=True)


def load_env_if_exists():
    if os.path.exists('.env'):
        load_dotenv('.env')