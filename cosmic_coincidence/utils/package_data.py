from pathlib import Path
from pkg_resources import resource_filename


def get_path_to_data(file_name: str) -> Path:

    file_path = resource_filename("cosmic_coincidence", "data/%s" % file_name)

    return Path(file_path)
