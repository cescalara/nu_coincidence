from popsynth.population_synth import PopulationSynth
from cosmic_coincidence.utils.parameter_server import ParameterServer


class PopsynthParams(ParameterServer):
    """
    Class to pass necessary params to create
    a popsynth population.
    """

    def __init__(self, popsynth_spec, flux_sigma=0.1):
        """
        :popsynth_spec: YAML file containing popsynth info.
        """

        super().__init__()

        self._popsynth_spec = popsynth_spec

        self._flux_sigma = flux_sigma

    @property
    def popsynth_spec(self):

        return self._popsynth_spec

    @property
    def flux_sigma(self):

        return self._flux_sigma


class PopsynthWrapper(object):
    """
    Wrapper to create popsynths from PopsynthParams.
    """

    def __init__(self, parameter_server):

        self._parameter_server = parameter_server

        ps = parameter_server.popsynth_spec

        fs = parameter_server.flux_sigma

        self._pop_gen = PopulationSynth.from_file(ps)

        self._pop_gen._seed = parameter_server.seed

        self._survey = self._pop_gen.draw_survey(flux_sigma=fs)

    @property
    def survey(self):

        return self._survey

    def write(self):

        pass
