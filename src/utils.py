import json
from urllib.parse import urlparse

import yaml


def is_url(url_or_filename: str) -> bool:
    """
    Checks if a given string is a valid URL.

    Args:
        url_or_filename (str): A string that may represent a URL.

    Returns:
        bool: True if the string is a valid URL, False otherwise.
    """
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")


def now():
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d%H%M")[:-1]


def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)


def load_yml(filename):
    with open(filename, "r") as f:
        return yaml.safe_load(f)


def load_tsv(filename) -> list[list[str]]:
    with open(filename, "r") as f:
        reader = csv.reader(f, delimiter='\t')
        data = [row for row in reader]
    return data
