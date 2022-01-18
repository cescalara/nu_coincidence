import pytest


@pytest.fixture(scope="session")
def output_directory(tmpdir_factory):

    directory = tmpdir_factory.mktemp("output")

    return directory
