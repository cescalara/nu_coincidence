from setuptools import setup
import versioneer
import os


def find_config_files(directory):

    paths = []

    for (path, directories, filenames) in os.walk(directory):

        for filename in filenames:

            paths.append(os.path.join("..", path, filename))

    return paths


config_files = find_config_files("nu_coincidence/config")

setup(
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    include_package_data=True,
    package_data={"": config_files},
)
