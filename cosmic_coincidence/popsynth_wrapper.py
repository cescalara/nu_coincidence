import abc


class PopsynthWrapper(object, metaclass=abc.ABCMeta):
    """
    Abstract base class showing how popsynths can be
    wrapped for parallel simulation
    """

    def __init__(self, parameter_server, serial=False):

        popsynth = self._pop_type(**parameter_server.parameters)

        survey = popsynth.draw_survey(**parameter_server.survey)

        survey.writeto(parameter_server.file_path)

        del survey

    def _pop_type(self, **kwargs):

        raise NotImplementedError()
