import numpy as np
import h5py

from ..populations.aux_samplers import (
    VariabilityAuxSampler,
    FlareRateAuxSampler,
    FlareTimeAuxSampler,
    FlareDurationAuxSampler,
)


def check_spatial_coincidence(nu_ra, nu_dec, nu_ang_err, pop):

    n_matches = []
    match_inds = []

    for r, d, err in zip(nu_ra, nu_dec, nu_ang_err):
        match = (pop.ra[pop.selection] - r) ** 2 + (
            pop.dec[pop.selection] - d
        ) ** 2 <= (err * 3) ** 2
        match_inds.append(np.where(match == True)[0])
        n_matches.append(len(match[match == True]))

    return n_matches, match_inds


def check_temporal_coincidence(nu_times, match_inds, pop):

    n_match_variable = 0
    n_match_flaring = 0

    for t, m in zip(nu_times, match_inds):

        if m.size:

            for i in m:

                if pop.variability_selected[i]:

                    n_match_variable += 1

                    fts = pop.flare_times_selected[i]
                    ds = pop.flare_durations_selected[i]

                    for f, d in zip(fts, ds):

                        if t >= f and t <= f + d:

                            n_match_flaring += 1

    return n_match_variable, n_match_flaring


def run_sim(
    bllac_ldde,
    fsrq_ldde,
    nu_simulator,
    seed,
    obs_time,
    Emin_det,
):

    bllac_popsynth = bllac_ldde.popsynth(seed=seed)
    variability = VariabilityAuxSampler()
    variability.weight = 0.05
    flare_rate = FlareRateAuxSampler()
    flare_rate.xmin = 1 / 7.5
    flare_rate.xmax = 15
    flare_rate.index = 1.5
    flare_times = FlareTimeAuxSampler()
    flare_times.obs_time = obs_time
    flare_durations = FlareDurationAuxSampler()
    flare_rate.set_secondary_sampler(variability)
    flare_times.set_secondary_sampler(flare_rate)
    flare_durations.set_secondary_sampler(flare_times)
    bllac_popsynth.add_observed_quantity(flare_durations)

    fsrq_popsynth = fsrq_ldde.popsynth(seed=seed)
    variability = VariabilityAuxSampler()
    variability.weight = 0.4
    flare_rate = FlareRateAuxSampler()
    flare_rate.xmin = 1 / 7.5
    flare_rate.xmax = 15
    flare_rate.index = 1.5
    flare_times = FlareTimeAuxSampler()
    flare_times.obs_time = obs_time
    flare_durations = FlareDurationAuxSampler()
    flare_rate.set_secondary_sampler(variability)
    flare_times.set_secondary_sampler(flare_rate)
    flare_durations.set_secondary_sampler(flare_times)
    fsrq_popsynth.add_observed_quantity(flare_durations)

    bllac_pop = bllac_popsynth.draw_survey(boundary=4e-12, hard_cut=True)
    fsrq_pop = fsrq_popsynth.draw_survey(boundary=4e-12, hard_cut=True)
    nu_simulator.run(show_progress=False, seed=seed)

    # Select neutrinos above reco energy threshold
    nu_selection = np.array(nu_simulator.reco_energy) > Emin_det
    nu_ra = np.rad2deg(nu_simulator.ra)[nu_selection]
    nu_dec = np.rad2deg(nu_simulator.dec)[nu_selection]
    nu_ang_err = np.array(nu_simulator.ang_err)[nu_selection]
    # nu_reco_energy = np.array(nu_simulator.reco_energy)[nu_selection]
    nu_times = np.random.uniform(0, obs_time, len(nu_ra))

    bllac_info = {}
    fsrq_info = {}

    # check coincidences in space and time
    # BL Lacs
    n_matches, match_inds = check_spatial_coincidence(
        nu_ra, nu_dec, nu_ang_err, bllac_pop
    )
    nv, nf = check_temporal_coincidence(nu_times, match_inds, bllac_pop)
    bllac_info["n_spatial"] = sum(n_matches)
    bllac_info["n_variable"] = nv
    bllac_info["n_flaring"] = nf

    # FSRQs
    n_matches, match_inds = check_spatial_coincidence(
        nu_ra, nu_dec, nu_ang_err, fsrq_pop
    )
    nv, nf = check_temporal_coincidence(nu_times, match_inds, fsrq_pop)
    fsrq_info["n_spatial"] = sum(n_matches)
    fsrq_info["n_variable"] = nv
    fsrq_info["n_flaring"] = nf

    return bllac_info, fsrq_info

    # Save useful info, appending for each round
    # with h5py.File(output_file, "r+") as f:

    #     f["bllac/n_spatial"][i_sim] = bllac_info["n_spatial"]
    #     f["bllac/n_variable"][i_sim] = bllac_info["n_variable"]
    #     f["bllac/n_flaring"][i_sim] = bllac_info["n_flaring"]

    #     f["fsrq/n_spatial"][i_sim] = fsrq_info["n_spatial"]
    #     f["fsrq/n_variable"][i_sim] = fsrq_info["n_variable"]
    #     f["fsrq/n_flaring"][i_sim] = fsrq_info["n_flaring"]


def submit_sim(args):

    (
        bllac_ldde,
        fsrq_ldde,
        nu_simulator,
        seed,
        i_sim,
        obs_time,
        Emin_det,
        output_file,
    ) = args

    run_sim(
        bllac_ldde,
        fsrq_ldde,
        nu_simulator,
        seed,
        i_sim,
        obs_time,
        Emin_det,
        output_file,
    )
