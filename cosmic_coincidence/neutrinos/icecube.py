import numpy as np
import h5py
import yaml
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

from icecube_tools.detector.effective_area import EffectiveArea
from icecube_tools.detector.energy_resolution import EnergyResolution
from icecube_tools.detector.angular_resolution import AngularResolution
from icecube_tools.detector.detector import IceCube
from icecube_tools.source.flux_model import PowerLawFlux
from icecube_tools.source.source_model import DiffuseSource, PointSource
from icecube_tools.simulator import Simulator

from popsynth.utils.cosmology import cosmology

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
    source_label: int
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

    @property
    def detector(self):

        return self._detector

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

        self._energy_res = EnergyResolution.from_dataset("20150820", fetch=False)

        self._hese_simulation_setup()

        self._ehe_simulation_setup()

    def _hese_simulation_setup(self):

        # Sources - all flavor flux
        hese_sources = []

        if self._parameter_server.hese.atmospheric_flux is not None:

            atmo_power_law = PowerLawFlux(
                **self._parameter_server.hese.atmospheric_flux
            )
            atmo_source = DiffuseSource(flux_model=atmo_power_law)
            hese_sources.append(atmo_source)

        if self._parameter_server.hese.diffuse_flux is not None:

            diffuse_power_law = PowerLawFlux(**self._parameter_server.hese.diffuse_flux)
            diffuse_source = DiffuseSource(flux_model=diffuse_power_law)
            hese_sources.append(diffuse_source)

        # Detector
        hese_aeff = EffectiveArea.from_dataset(
            "20131121",
            scale_factor=0.12,
            fetch=False,
        )

        hese_ang_res = AngularResolution.from_dataset(
            "20181018",
            ret_ang_err_p=0.9,
            offset=-0.2,
            scale=3,
            scatter=0.5,
            minimum=0.2,
            fetch=False,
        )

        self._hese_detector = IceCube(hese_aeff, self._energy_res, hese_ang_res)

        self._hese_simulator = Simulator(hese_sources, self._hese_detector)
        self._hese_simulator.time = self._parameter_server.hese.detector["obs_time"]
        self._hese_simulator.max_cosz = self._parameter_server.hese.detector["max_cosz"]

    def _ehe_simulation_setup(self):

        # Sources - only numu flux
        ehe_sources = []

        if self._parameter_server.ehe.atmospheric_flux is not None:

            atmo_power_law = PowerLawFlux(**self._parameter_server.ehe.atmospheric_flux)
            atmo_source = DiffuseSource(flux_model=atmo_power_law)
            ehe_sources.append(atmo_source)

        if self._parameter_server.ehe.diffuse_flux is not None:

            diffuse_power_law = PowerLawFlux(**self._parameter_server.ehe.diffuse_flux)
            diffuse_source = DiffuseSource(flux_model=diffuse_power_law)
            ehe_sources.append(diffuse_source)

        ehe_aeff = EffectiveArea.from_dataset("20181018", fetch=False)

        ehe_ang_res = AngularResolution.from_dataset(
            "20181018",
            ret_ang_err_p=0.9,
            offset=0.0,
            scale=1,
            minimum=0.2,
            scatter=0.2,
            fetch=False,
        )

        self._ehe_detector = IceCube(ehe_aeff, self._energy_res, ehe_ang_res)

        self._ehe_simulator = Simulator(ehe_sources, self._ehe_detector)
        self._ehe_simulator.time = self._parameter_server.ehe.detector["obs_time"]
        self._ehe_simulator.max_cosz = self._parameter_server.ehe.detector["max_cosz"]

    def _run(self):

        # Only run independent sim if neutrinos are not
        # connected to another population.

        if (
            self._parameter_server.hese.connection is None
            and self._parameter_server.ehe.connection is None
        ):

            # HESE
            self._hese_simulator.run(
                show_progress=False, seed=self._parameter_server.seed
            )

            hese_Emin_det = self._parameter_server.hese.detector["Emin_det"]
            hese_selection = np.array(self._hese_simulator.reco_energy) > hese_Emin_det

            hese_times = np.random.uniform(
                0,
                self._parameter_server.hese.detector["obs_time"],
                self._hese_simulator.N,
            )

            # EHE
            self._ehe_simulator.run(
                show_progress=False, seed=self._parameter_server.seed
            )

            ehe_Emin_det = self._parameter_server.ehe.detector["Emin_det"]
            ehe_selection = np.array(self._ehe_simulator.reco_energy) > ehe_Emin_det

            ehe_times = np.random.uniform(
                0,
                self._parameter_server.ehe.detector["obs_time"],
                self._ehe_simulator.N,
            )

            # Combine
            selection = np.concatenate((hese_selection, ehe_selection))

            ra = np.concatenate((self._hese_simulator.ra, self._ehe_simulator.ra))
            ra = np.rad2deg(ra[selection])

            dec = np.concatenate((self._hese_simulator.dec, self._ehe_simulator.dec))
            dec = np.rad2deg(dec[selection])

            ang_err = np.concatenate(
                (self._hese_simulator.ang_err, self._ehe_simulator.ang_err)
            )
            ang_err = ang_err[selection]

            energies = np.concatenate(
                (self._hese_simulator.reco_energy, self._ehe_simulator.reco_energy)
            )
            energies = energies[selection]

            source_labels = np.concatenate(
                (self._hese_simulator.source_label, self._ehe_simulator.source_label)
            )
            source_labels = source_labels[selection]

            times = np.concatenate((hese_times, ehe_times))[selection]

            self._observation = IceCubeObservation(
                energies,
                ra,
                dec,
                ang_err,
                times,
                selection,
                source_labels,
            )

        else:

            self._observation = None


class IceCubeTracksWrapper(IceCubeObsWrapper):
    """
    Wrapper for the simulation of track events.
    """

    def __init__(self, param_server):

        super().__init__(param_server)

    def _simulation_setup(self):

        # Sources
        sources = []

        if self._parameter_server.atmospheric_flux is not None:

            atmo_power_law = PowerLawFlux(**self._parameter_server.atmospheric_flux)
            atmo_source = DiffuseSource(flux_model=atmo_power_law)
            sources.append(atmo_source)

        if self._parameter_server.diffuse_flux is not None:

            diffuse_power_law = PowerLawFlux(**self._parameter_server.diffuse_flux)
            diffuse_source = DiffuseSource(flux_model=diffuse_power_law)
            sources.append(diffuse_source)

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

        self._detector = IceCube(
            effective_area,
            energy_resolution,
            angular_resolution,
        )

        self._simulator = Simulator(sources, self._detector)
        self._simulator.time = self._parameter_server.detector["obs_time"]
        self._simulator.max_cosz = self._parameter_server.detector["max_cosz"]

    def _run(self):

        # Only run independent simulation if neutrinos are not
        # connected to another population

        if self._parameter_server.connection is None:

            self._simulator.run(
                show_progress=False,
                seed=self._parameter_server.seed,
            )

            # Select neutrinos above reco energy threshold
            Emin_det = self._parameter_server.detector["Emin_det"]
            selection = np.array(self._simulator.reco_energy) > Emin_det
            ra = np.rad2deg(self._simulator.ra)[selection]
            dec = np.rad2deg(self._simulator.dec)[selection]
            ang_err = np.array(self._simulator.ang_err)[selection]
            energies = np.array(self._simulator.reco_energy)[selection]
            source_labels = np.array(self._simulator.source_label)[selection]
            times = np.random.uniform(
                0,
                self._parameter_server.detector["obs_time"],
                self._simulator.N,
            )

            self._observation = IceCubeObservation(
                energies,
                ra,
                dec,
                ang_err,
                times,
                selection,
                source_labels,
            )

        else:

            self._observation = None


class IceCubeObsParams(ParameterServer):
    """
    All the info you need to recreate a
    simulation of neutrinos in IceCube.
    """

    def __init__(
        self,
        detector: Dict[str, Any],
        atmospheric_flux: Optional[Dict[str, Any]] = None,
        diffuse_flux: Optional[Dict[str, Any]] = None,
        connection: Optional[Dict[str, Any]] = None,
    ):

        super().__init__()

        self._detector = detector

        self._atmospheric_flux = atmospheric_flux

        self._diffuse_flux = diffuse_flux

        self._connection = connection

    def to_dict(self) -> Dict[str, Any]:

        output: Dict[str, Any] = {}

        output["detector"] = self._detector

        if self._atmospheric_flux is not None:

            output["atmospheric flux"] = self._atmospheric_flux

        if self._diffuse_flux is not None:

            output["diffuse flux"] = self._diffuse_flux

        if self._connection is not None:

            output["connection"] = self._connection

        return output

    @classmethod
    def from_dict(cls, input: Dict[str, Any]) -> "IceCubeObsParams":

        detector = input["detector"]

        if "atmospheric flux" in input:

            atmospheric_flux = input["atmospheric flux"]

        else:

            atmospheric_flux = None

        if "diffuse flux" in input:

            diffuse_flux = input["diffuse flux"]

        else:

            diffuse_flux = None

        if "connection" in input:

            connection = input["connection"]

        else:

            connection = None

        return cls(detector, atmospheric_flux, diffuse_flux, connection)

    def write_to(self, file_name: str):

        with open(file_name, "w") as f:

            yaml.dump(
                stream=f,
                data=self.to_dict(),
                # default_flow_style=False,
                Dumper=yaml.SafeDumper,
            )

    @classmethod
    def from_file(cls, file_name: str) -> "IceCubeObsParams":

        with open(file_name) as f:

            input: Dict[str, Any] = yaml.load(f, Loader=yaml.SafeLoader)

        return cls.from_dict(input)

    @property
    def detector(self):

        return self._detector

    @property
    def atmospheric_flux(self):

        return self._atmospheric_flux

    @property
    def diffuse_flux(self):

        return self._diffuse_flux

    @property
    def connection(self):

        return self._connection


class IceCubeAlertsParams(ParameterServer):
    """
    Parameter server for IceCube alerts where
    HESE and EHE simulations both require inputs.

    For use with IceCubeAlertsWrapper
    """

    def __init__(
        self,
        hese_config_file: str,
        ehe_config_file: str,
    ):

        super().__init__()

        self._hese = IceCubeObsParams.from_file(hese_config_file)

        self._ehe = IceCubeObsParams.from_file(ehe_config_file)

    @property
    def hese(self):

        return self._hese

    @property
    def ehe(self):

        return self._ehe


def _get_point_source(
    luminosity,
    spectral_index,
    z,
    ra,
    dec,
    Emin,
    Emax,
    Enorm,
):
    """
    Define a neutrino point source from
    a luminosity and spectral index.

    :param luminosity: L in GeV s^-1
    :param spectral_index: Spectral index of power law
    :param z: Redshift
    :param ra: Right ascension
    :param dec: Declination
    :param Emin: Minimum energy in GeV
    :param Emax: Maximum energy in GeV
    :param Enorm: Normalisation energy in GeV
    """

    energy_flux = luminosity / (4 * np.pi * cosmology.luminosity_distance(z) ** 2)

    tmp = PowerLawFlux(
        1,
        Enorm,
        spectral_index,
        lower_energy=Emin,
        upper_energy=Emax,
    )

    power = tmp.total_flux_density()

    norm = energy_flux / power

    power_law = PowerLawFlux(
        norm, Enorm, spectral_index, lower_energy=Emin, upper_energy=Emax
    )

    source = PointSource(flux_model=power_law, coord=(ra, dec))

    return source


def _run_sim_for(
    N,
    spectral_index,
    ra,
    dec,
    Emin,
    Emax,
    Enorm,
    detector,
    seed,
):
    """
    Run a simulation of N events with the provided
    spectral model and detector info.

    :param N: Integer number of neutrinos
    :param spectral_index: Spectral index of power law
    :param ra: Right ascension
    :param dec: Declination
    :param Emin: Minimum energy in GeV
    :param Emax: Maximum energy in GeV
    :param Enorm: Normalisation energy in GeV
    :param Detector: IceCube detector
    :param seed: Random seed
    """

    tmp = PowerLawFlux(1, Enorm, spectral_index, lower_energy=Emin, upper_energy=Emax)

    source = PointSource(flux_model=tmp, coord=(ra, dec))

    sim = Simulator(source, detector)

    sim.run(
        N=N,
        show_progress=False,
        seed=seed,
    )

    return sim
