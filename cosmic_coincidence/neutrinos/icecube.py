import numpy as np
import h5py
from abc import ABCMeta, abstractmethod
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
    source_component: int
    name: str = "icecube_obs"


class IceCubeObsWrapper(object, metaclass=ABCMeta):
    """
    Abstract base class for IceCube-like
    observations.
    """

    def __init__(self, param_server):

        self._parameter_server = param_server

        self._simulation_setup()

        self._run()

    @abstractmethod
    def _simulation_setup(self):

        pass

    @abstractmethod
    def _run(self):

        pass

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


class IceCubeAlertsWrapper(IceCubeObsWrapper):
    """
    Wrapper for simulation of HESE and EHE
    alerts.
    """

    def __init__(self, param_server):

        super().__init__(param_server)

    def _simulation_setup(self):

        self._energy_res = EnergyResolution.from_dataset("20150820")

        self._hese_simulation_setup()

        self._ehe_simulation_setup()

    def _hese_simulation_setup(self):

        # Sources - all flavor flux
        atmo_power_law = PowerLawFlux(**self._parameter_server.hese.atmospheric)
        atmo_source = DiffuseSource(flux_model=atmo_power_law)

        diffuse_power_law = PowerLawFlux(**self._parameter_server.hese.diffuse)
        diffuse_source = DiffuseSource(flux_model=diffuse_power_law)

        hese_sources = [atmo_source, diffuse_source]

        # Detector
        hese_aeff = EffectiveArea.from_dataset("20131121", scale_factor=0.12)

        hese_ang_res = AngularResolution.from_dataset(
            "20181018",
            ret_ang_err_p=0.9,
            offset=-0.2,
            scale=3,
            scatter=0.5,
            minimum=0.2,
        )

        hese_detector = IceCube(hese_aeff, self._energy_res, hese_ang_res)

        self._hese_simulator = Simulator(hese_sources, hese_detector)
        self._hese_simulator.time = self._parameter_server.hese.obs_time
        self._hese_simulator.max_cosz = self._parameter_server.hese.max_cosz

    def _ehe_simulation_setup(self):

        # Sources - only numu flux
        atmo_power_law = PowerLawFlux(**self._parameter_server.ehe.atmospheric)
        atmo_source = DiffuseSource(flux_model=atmo_power_law)

        diffuse_power_law = PowerLawFlux(**self._parameter_server.ehe.diffuse)
        diffuse_source = DiffuseSource(flux_model=diffuse_power_law)

        ehe_sources = [atmo_source, diffuse_source]

        ehe_aeff = EffectiveArea.from_dataset("20181018")

        ehe_ang_res = AngularResolution.from_dataset(
            "20181018",
            ret_ang_err_p=0.9,
            offset=0.0,
            scale=1,
            minimum=0.2,
            scatter=0.2,
        )

        ehe_detector = IceCube(ehe_aeff, self._energy_res, ehe_ang_res)

        self._ehe_detector = Simulator(ehe_sources, ehe_detector)
        self._ehe_simulator.time = self._parameter_server.ehe.obs_time
        self._ehe_simulator.max_cosz = self._parameter_server.ehe.max_cosz

    def _run(self):

        pass


class IceCubeTracksWrapper(IceCubeObsWrapper):
    """
    Wrapper for the simulation of track events.
    """

    def __init__(self, param_server):

        super().__init__(param_server)

    def _simulation_setup(self):

        # Sources
        atmo_power_law = PowerLawFlux(**self._parameter_server.atmospheric)
        atmo_source = DiffuseSource(flux_model=atmo_power_law)

        diffuse_power_law = PowerLawFlux(**self._parameter_server.diffuse)
        diffuse_source = DiffuseSource(flux_model=diffuse_power_law)

        sources = [atmo_source, diffuse_source]

        # Detector
        effective_area = EffectiveArea.from_dataset("20181018", fetch=False)

        angular_resolution = AngularResolution.from_dataset(
            "20181018",
            fetch=False,
            ret_ang_err_p=0.9,  # Return 90% CIs
            offset=0.4,  # Shift angular error up in attempt to match HESE
        )

        energy_resolution = EnergyResolution.from_dataset(
            "20150820",
            fetch=False,
        )

        detector = IceCube(
            effective_area,
            energy_resolution,
            angular_resolution,
        )

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


class IceCubeAlertParams(ParameterServer):
    """
    Parameter server for IceCube alerts where
    HESE and EHE simulations both require inputs.
    """

    def __init__(
        self,
        hese_Emin,
        ehe_Emin,
        Emax,
        Enorm,
        hese_Emin_det,
        ehe_Emin_det,
        hese_atmo_flux_norm,
        ehe_atmo_flux_norm,
        atmo_index,
        hese_diff_flux_norm,
        ehe_diff_flux_norm,
        diff_index,
        max_cosz,
        obs_time,
    ):

        super().__init__()

        self._hese = IceCubeObsParams(
            hese_Emin,
            Emax,
            Enorm,
            hese_Emin_det,
            hese_atmo_flux_norm,
            atmo_index,
            hese_diff_flux_norm,
            diff_index,
            max_cosz,
            obs_time,
        )

        self._ehe = IceCubeObsParams(
            ehe_Emin,
            Emax,
            Enorm,
            ehe_Emin_det,
            ehe_atmo_flux_norm,
            atmo_index,
            ehe_diff_flux_norm,
            diff_index,
            max_cosz,
            obs_time,
        )

    @property
    def hese(self):

        return self._hese

    @property
    def ehe(self):

        return self._ehe
