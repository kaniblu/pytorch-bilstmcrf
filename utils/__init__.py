import os


def ensure_dir_exists(path):
    dir = os.path.dirname(path)
    os.makedirs(dir, exist_ok=True)