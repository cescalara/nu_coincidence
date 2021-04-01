import numpy as np
import h5py
from dataclasses import dataclass

from icecube_tools.detector.effective_area import EffectiveArea
from icecube_tools.detector.energy_resolution import EnergyResolution
from icecube_tools.detector.angular_resolution import AngularResolution
from icecube_tools.detector.detector import IceCube
from icecube_tools.source.flux_model import PowerLawFlux
from icecube_tools.source.source_model import DiffuseSource
from icecube_tools.simulator import Simulator

from cosmic_coincidence.utils.parameter_server import ParameterServer


@dataclass
class IceCubeObservation(object):
    """
    Store the output of IceCube simulations.
    """

    energies: float
    ra: float
    dec: float
    ang_err: float
    times: float
    selection: bool
    name: str = "icecube_obs"


class IceCubeObsWrapper(object):
    """
    Wrapper for the simulation of IceCube observations.
    """

    def __init__(self, param_server):

        self._parameter_server = param_server

        self._simulation_setup()

        self._run()

    def _simulation_setup(self):

        effective_area = EffectiveArea.from_dataset("20181018")

        angular_resolution = AngularResolution.from_dataset(
            "20181018",
            ret_ang_err_p=0.9,  # Return 90% CIs
            offset=0.4,  # Shift angular error up in attempt to match HESE
        )

        energy_resolution = EnergyResolution.from_dataset("20150820")

        detector = IceCube(
            effective_area,
            energy_resolution,
            angular_resolution,
        )

        atmo_power_law = PowerLawFlux(**self._parameter_server.atmospheric)
        atmo_source = DiffuseSource(flux_model=atmo_power_law)

        diffuse_power_law = PowerLawFlux(**self._parameter_server.diffuse)
        diffuse_source = DiffuseSource(flux_model=diffuse_power_law)

        sources = [atmo_source, diffuse_source]

        self._simulator = Simulator(sources, detector)
        self._simulator.time = self._parameter_server.obs_time
        self._simulator.max_cosz = self._parameter_server.max_cosz

    def _run(self):

        self._simulator.run(
            show_progress=False,
            seed=self._parameter_server.seed,
        )

        # Select neutrinos above reco energy threshold
        Emin_det = self._parameter_server.Emin_det
        selection = np.array(self._simulator.reco_energy) > Emin_det
        ra = np.rad2deg(self._simulator.ra)[selection]
        dec = np.rad2deg(self._simulator.dec)[selection]
        ang_err = np.array(self._simulator.ang_err)[selection]
        energies = np.array(self._simulator.reco_energy)[selection]
        N = len(ra)
        times = np.random.uniform(0, self._parameter_server.obs_time, N)

        self._observation = IceCubeObservation(
            energies, ra, dec, ang_err, times, selection
        )

    @property
    def observation(self):

        return self._observation

    def write(self):

        with h5py.File(self._parameter_server.file_name, "r+") as f:

            if self._parameter_server.group_name not in f.keys():

                group = f.create_group(self._parameter_server.group_name)

            else:

                group = f[self._parameter_server.group_name]

            subgroup = group.create_group(self._observation.name)

            for key, value in vars(self._observation).items():

                if key != "name" and key != "selection":

                    subgroup.create_dataset(key, data=value, compression="lzf")

            for key, value in self._parameter_server.parameters.items():

                subgroup.create_dataset(key, data=value)


class IceCubeObsParams(ParameterServer):
    """
    Parameter server for simulations of
    IceCube observations.
    """

    def __init__(
        self,
        Emin,
        Emax,
        Enorm,
        Emin_det,
        atmo_flux_norm,
        atmo_index,
        diff_flux_norm,
        diff_index,
        max_cosz,
        obs_time,
    ):

        super().__init__()

        self._parameters = dict(
            Emin=Emin,
            Emax=Emax,
            Enorm=Enorm,
            Emin_det=Emin_det,
            max_cosz=max_cosz,
            obs_time=obs_time,
        )

        self._atmospheric = dict(
            normalisation=atmo_flux_norm,
            normalisation_energy=Enorm,
            index=atmo_index,
            lower_energy=Emin,
            upper_energy=Emax,
        )

        self._diffuse = dict(
            normalisation=diff_flux_norm,
            normalisation_energy=Enorm,
            index=diff_index,
            lower_energy=Emin,
            upper_energy=Emax,
        )

        self._obs_time = obs_time

        self._max_cosz = max_cosz

        self._Emin_det = Emin_det

    @property
    def atmospheric(self):

        return self._atmospheric

    @property
    def diffuse(self):

        return self._diffuse

    @property
    def obs_time(self):

        return self._obs_time

    @property
    def max_cosz(self):

        return self._max_cosz

    @property
    def Emin_det(self):

        return self._Emin_det
