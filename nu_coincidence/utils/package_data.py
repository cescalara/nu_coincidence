from pathlib import Path
from pkg_resources import resource_filename


def get_path_to_config(file_name: str) -> Path:

    file_path = resource_filename("nu_coincidence", "config/%s" % file_name)

    return Path(file_path)
