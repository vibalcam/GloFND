import json
import os
import click

def colorful_print(string: str, *args, **kwargs) -> None:
    print(click.style(string, *args, **kwargs))

def load_json(file_path):
    """Load a JSON file from the given file path."""
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def save_json(data, file_path):
    """Save a JSON object to a given file path."""
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


def append_to_json(data, file_path):
    """Append a JSON object to a given file path."""
    # check if file exists
    if os.path.exists(file_path):
        old_data = load_json(file_path)
    else:
        old_data = []
    if not isinstance(old_data, list):
        old_data = [old_data]
    old_data.append(data)
    with open(file_path, 'w') as file:
        json.dump(old_data, file, indent=4)
