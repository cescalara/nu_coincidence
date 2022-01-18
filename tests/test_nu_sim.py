from icecube_tools.detector.effective_area import EffectiveArea
from icecube_tools.detector.energy_resolution import EnergyResolution
from icecube_tools.detector.angular_resolution import AngularResolution

from nu_coincidence.utils.package_data import (
    get_available_config,
    get_path_to_config,
)
from nu_coincidence.neutrinos.icecube import (
    IceCubeAlertsWrapper,
    IceCubeAlertsParams,
    IceCubeTracksWrapper,
    IceCubeObsParams,
)

config_files = get_available_config()

# Make sure icecube_tools data is loaded
my_aeff = EffectiveArea.from_dataset("20181018")
my_aeff = EffectiveArea.from_dataset("20131121")
my_eres = EnergyResolution.from_dataset("20150820")
my_angres = AngularResolution.from_dataset("20181018")


def test_icecube_alerts_diffuse_sim():

    hese_config_files = [f for f in config_files if ("hese" in f and "diffuse" in f)]

    ehe_config_files = [f for f in config_files if ("ehe" in f and "diffuse" in f)]

    for hese_config_file, ehe_config_file in zip(
        hese_config_files,
        ehe_config_files,
    ):

        print(hese_config_file, ehe_config_file)

        hese_config = get_path_to_config(hese_config_file)

        ehe_config = get_path_to_config(ehe_config_file)

        param_server = IceCubeAlertsParams(hese_config, ehe_config)

        param_server.seed = 42

        nu_obs = IceCubeAlertsWrapper(param_server)

        assert len(nu_obs.observation.ra) == len(nu_obs.observation.dec)

        assert len(nu_obs.observation.ra) > 0


def test_icecube_tracks_diffuse_sim():

    track_config_files = [f for f in config_files if ("tracks" in f and "diffuse" in f)]

    for track_config_file in track_config_files:

        track_config = get_path_to_config(track_config_file)

        param_server = IceCubeObsParams.from_file(track_config)

        param_server.seed = 42

        nu_obs = IceCubeTracksWrapper(param_server)

        assert len(nu_obs.observation.ra) == len(nu_obs.observation.dec)

        assert len(nu_obs.observation.ra) > 0
