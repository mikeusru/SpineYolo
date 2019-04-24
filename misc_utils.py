
import os


def format_path_for_os(old_path):
    new_path = old_path.replace('//', os.sep).replace('/', os.sep).replace('\\', os.sep)
    return new_path

