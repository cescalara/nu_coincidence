from abc import ABCMeta


class ParameterServer(object, metaclass=ABCMeta):
    """
    Abstract base class for parameter servers.
    """

    def __init__(self):

        self._file_name = None

        self._group_name = None

        self._seed = None

        self._parameters = None

    @property
    def seed(self):

        return self._seed

    @seed.setter
    def seed(self, value: int):

        self._seed = value

    @property
    def file_name(self):

        return self._file_name

    @file_name.setter
    def file_name(self, value: str):

        self._file_name = value

    @property
    def group_name(self):

        return self._group_name

    @group_name.setter
    def group_name(self, value: str):

        self._group_name = value

    @property
    def parameters(self):

        return self._parameters
