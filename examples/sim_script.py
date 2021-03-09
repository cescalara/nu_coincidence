import numpy as np
import h5py
from mpi4py import MPI
import warnings
import time

from popsynth.utils.configuration import popsynth_config

import sys

sys.path.append("../")
from cosmic_coincidence.utils.interface import BLLacLDDEModel, FSRQLDDEModel
from cosmic_coincidence.utils.plotting import SphericalCircle
from cosmic_coincidence.utils.coincidence import (
    check_spatial_coincidence,
    check_temporal_coincidence,
    run_sim,
    submit_sim,
)

from icecube_tools.detector.effective_area import EffectiveArea
from icecube_tools.detector.energy_resolution import EnergyResolution
from icecube_tools.detector.angular_resolution import AngularResolution
from icecube_tools.detector.detector import IceCube
from icecube_tools.source.flux_model import PowerLawFlux, BrokenPowerLawFlux
from icecube_tools.source.source_model import DiffuseSource
from icecube_tools.simulator import Simulator

COMM = MPI.COMM_WORLD

popsynth_config["show_progress"] = False
warnings.filterwarnings("ignore")

if COMM.rank == 0:

    obs_time = 10  # years
    N = 1000
    output_file = "output/test_sim_err2_1000.h5"

    trial = range(N)
    trials = np.array_split(trial, COMM.size)
    # print(str(N_per_core) + " runs on each core")
    print("Setup...")

    popsynth_config["show_progress"] = False

    with h5py.File(output_file, "w") as f:
        f.create_dataset("obs_time", data=obs_time)
        f.create_dataset("N", data=N)

    # Blazar models
    bllac_ldde = BLLacLDDEModel()
    bllac_ldde.A = 3.39e4
    bllac_ldde.gamma1 = 0.27
    bllac_ldde.Lstar = 0.28e48
    bllac_ldde.gamma2 = 1.86
    bllac_ldde.zcstar = 1.34
    bllac_ldde.p1star = 2.24
    bllac_ldde.tau = 4.92
    bllac_ldde.p2 = -7.37
    bllac_ldde.alpha = 4.53e-2
    bllac_ldde.mustar = 2.1
    bllac_ldde.beta = 6.46e-2
    bllac_ldde.sigma = 0.26
    bllac_ldde.Lmax = 1e50
    bllac_ldde.prep_pop()

    fsrq_ldde = FSRQLDDEModel()
    fsrq_ldde.A = 3.06e4
    fsrq_ldde.gamma1 = 0.21
    fsrq_ldde.Lstar = 0.84e48
    fsrq_ldde.gamma2 = 1.58
    fsrq_ldde.zcstar = 1.47
    fsrq_ldde.p1star = 7.35
    fsrq_ldde.tau = 0
    fsrq_ldde.p2 = -6.51
    fsrq_ldde.alpha = 0.21
    fsrq_ldde.mustar = 2.44
    fsrq_ldde.beta = 0
    fsrq_ldde.sigma = 0.18
    fsrq_ldde.Lmax = 1e50
    fsrq_ldde.prep_pop()

    # Neutrino model
    Emin = 1e5  # GeV
    Emax = 1e8  # GeV
    Emin_det = 2e5  # GeV

    # Effective area
    Aeff_filename = "input/IC86-2012-TabulatedAeff.txt"
    effective_area = EffectiveArea(Aeff_filename)

    # Energy resolution
    eres_file = "input/effective_area.h5"
    energy_res = EnergyResolution(eres_file)

    # Angular resolution
    Ares_file = "input/IC86-2012-AngRes.txt"
    ang_res = AngularResolution(Ares_file)

    # Detector
    detector = IceCube(effective_area, energy_res, ang_res)

    power_law_atmo = PowerLawFlux(
        2.5e-18, 1e5, 3.7, lower_energy=Emin, upper_energy=1e8
    )
    atmospheric = DiffuseSource(flux_model=power_law_atmo)
    power_law = PowerLawFlux(1.0e-18, 1e5, 2.19, lower_energy=Emin, upper_energy=1e8)
    astrophysical_bg = DiffuseSource(flux_model=power_law)
    sources = [atmospheric, astrophysical_bg]

    nu_simulator = Simulator(sources, detector)
    nu_simulator.time = obs_time  # years
    nu_simulator.max_cosz = 0.1

    print("Starting...")
    start_time = time.time()

else:

    bllac_ldde = None
    fsrq_ldde = None
    nu_simulator = None
    obs_time = None
    Emin_det = None
    trials = None

bllac_ldde = COMM.bcast(bllac_ldde, root=0)
fsrq_ldde = COMM.bcast(fsrq_ldde, root=0)
nu_simulator = COMM.bcast(nu_simulator, root=0)
obs_time = COMM.bcast(obs_time, root=0)
Emin_det = COMM.bcast(Emin_det, root=0)
trials = COMM.scatter(trials, root=0)

output = []

for i in trials:

    # print(i)

    seed = int(i * 10)

    bllac_info, fsrq_info = run_sim(
        bllac_ldde,
        fsrq_ldde,
        nu_simulator,
        seed,
        obs_time,
        Emin_det,
    )

    output.append((bllac_info, fsrq_info))

outputs = MPI.COMM_WORLD.gather(output, root=0)

if COMM.rank == 0:

    print("Done! Time:", time.time() - start_time)
    print("Saving...")

    bllac_n_spatial = []
    bllac_n_variable = []
    bllac_n_flaring = []
    fsrq_n_spatial = []
    fsrq_n_variable = []
    fsrq_n_flaring = []

    for output in outputs:
        for o in output:
            bllac_info, fsrq_info = o
            bllac_n_spatial.append(bllac_info["n_spatial"])
            bllac_n_variable.append(bllac_info["n_variable"])
            bllac_n_flaring.append(bllac_info["n_flaring"])
            fsrq_n_spatial.append(fsrq_info["n_spatial"])
            fsrq_n_variable.append(fsrq_info["n_variable"])
            fsrq_n_flaring.append(fsrq_info["n_flaring"])

    # gather results and save
    with h5py.File(output_file, "r+") as f:
        bllac = f.create_group("bllac")
        fsrq = f.create_group("fsrq")
        bllac.create_dataset("n_spatial", data=bllac_n_spatial)
        bllac.create_dataset("n_variable", data=bllac_n_variable)
        bllac.create_dataset("n_flaring", data=bllac_n_flaring)
        fsrq.create_dataset("n_spatial", data=fsrq_n_spatial)
        fsrq.create_dataset("n_variable", data=fsrq_n_variable)
        fsrq.create_dataset("n_flaring", data=fsrq_n_flaring)
