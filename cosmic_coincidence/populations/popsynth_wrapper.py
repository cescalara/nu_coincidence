import yaml
from typing import Dict, Any

from popsynth.population_synth import PopulationSynth
from cosmic_coincidence.utils.parameter_server import ParameterServer


class PopsynthParams(ParameterServer):
    """
    Class to pass necessary params to create
    a popsynth population.
    """

    def __init__(self, config_file, flux_sigma=0.1):
        """
        :popsynth_spec: YAML file containing popsynth info.
        """

        super().__init__()

        self._config_file = config_file

        self._flux_sigma = flux_sigma

        with open(self._config_file) as f:

            self._pop_spec: Dict[str, Any] = yaml.load(f, Loader=yaml.SafeLoader)

    @property
    def pop_spec(self):

        return self._pop_spec

    @property
    def flux_sigma(self):

        return self._flux_sigma


class PopsynthWrapper(object):
    """
    Wrapper to create popsynths from PopsynthParams.
    """

    def __init__(self, parameter_server):

        self._parameter_server = parameter_server

        ps = parameter_server.pop_spec

        fs = parameter_server.flux_sigma

        self._pop_gen = PopulationSynth.from_dict(ps)

        self._pop_gen._seed = parameter_server.seed

        self._survey = self._pop_gen.draw_survey(flux_sigma=fs)

    @property
    def survey(self):

        return self._survey

    def write(self):

        pass
