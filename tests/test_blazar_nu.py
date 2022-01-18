import time

from nu_coincidence.blazar_nu.coincidence import BlazarNuCoincidenceSim
from nu_coincidence.blazar_nu.coincidence import BlazarNuCoincidenceResults
from nu_coincidence.blazar_nu.connected import BlazarNuConnectedSim
from nu_coincidence.blazar_nu.connected import BlazarNuConnectedResults


coincidence_sim_params = {}
coincidence_sim_params["bllac_config"] = "bllac_ref.yml"
coincidence_sim_params["fsrq_config"] = "fsrq_ref.yml"
coincidence_sim_params["nu_hese_config"] = "nu_diffuse_hese.yml"
coincidence_sim_params["nu_ehe_config"] = "nu_diffuse_ehe.yml"
coincidence_sim_params["seed"] = 42

connected_sim_params = {}
connected_sim_params["flux_factors"] = [0.001, 0.01]
connected_sim_params["bllac_config"] = "bllac_ref.yml"
connected_sim_params["fsrq_config"] = "fsrq_ref.yml"
connected_sim_params["nu_hese_config"] = "nu_connected_hese.yml"
connected_sim_params["nu_ehe_config"] = "nu_connected_ehe.yml"
connected_sim_params["seed"] = 42


def test_concidence_sim(output_directory):

    sim = BlazarNuCoincidenceSim(
        file_name=output_directory.join("test_coincidence_sim.h5"),
        N=1,
        bllac_config=coincidence_sim_params["bllac_config"],
        fsrq_config=coincidence_sim_params["fsrq_config"],
        nu_hese_config=coincidence_sim_params["nu_hese_config"],
        nu_ehe_config=coincidence_sim_params["nu_ehe_config"],
        seed=coincidence_sim_params["seed"],
    )

    sim.run(parallel=False)

    results = BlazarNuCoincidenceResults.load(
        [output_directory.join("test_coincidence_sim.h5")]
    )

    assert len(results.bllac["n_spatial"]) == len(results.fsrq["n_spatial"]) == 1


def test_coincidence_sim_parallel(output_directory):

    sim = BlazarNuCoincidenceSim(
        file_name=output_directory.join("test_coincidence_sim_parallel.h5"),
        N=2,
        bllac_config=coincidence_sim_params["bllac_config"],
        fsrq_config=coincidence_sim_params["fsrq_config"],
        nu_hese_config=coincidence_sim_params["nu_hese_config"],
        nu_ehe_config=coincidence_sim_params["nu_ehe_config"],
        seed=coincidence_sim_params["seed"],
    )

    sim.run(parallel=True, n_jobs=2)

    time.sleep(1)

    results = BlazarNuCoincidenceResults.load(
        [output_directory.join("test_coincidence_sim_parallel.h5")]
    )

    assert len(results.bllac["n_spatial"]) == len(results.fsrq["n_spatial"]) == 2


def test_connected_sim(output_directory):

    sub_file_names = [
        output_directory.join("test_connected_sim_%.1e.h5" % ff)
        for ff in connected_sim_params["flux_factors"]
    ]

    for ff, sfn in zip(connected_sim_params["flux_factors"], sub_file_names):

        sim = BlazarNuConnectedSim(
            file_name=sfn,
            N=1,
            bllac_config=connected_sim_params["bllac_config"],
            fsrq_config=connected_sim_params["fsrq_config"],
            nu_hese_config=connected_sim_params["nu_hese_config"],
            nu_ehe_config=connected_sim_params["nu_ehe_config"],
            seed=connected_sim_params["seed"],
            flux_factor=ff,
            flare_only=True,
            det_only=True,
        )

        sim.run(parallel=False)

    BlazarNuConnectedResults.merge_over_flux_factor(
        sub_file_names,
        connected_sim_params["flux_factors"],
        write_to=output_directory.join("test_connected_sim.h5"),
    )

    results = BlazarNuConnectedResults.load(
        [output_directory.join("test_connected_sim.h5")]
    )

    assert len(results.bllac["n_alerts"]) == len(results.fsrq["n_alerts"]) == 2


def test_connected_sim_parallel(output_directory):

    sub_file_names = [
        output_directory.join("test_connected_sim_%.1e.h5" % ff)
        for ff in connected_sim_params["flux_factors"]
    ]

    for ff, sfn in zip(connected_sim_params["flux_factors"], sub_file_names):

        sim = BlazarNuConnectedSim(
            file_name=sfn,
            N=2,
            bllac_config=connected_sim_params["bllac_config"],
            fsrq_config=connected_sim_params["fsrq_config"],
            nu_hese_config=connected_sim_params["nu_hese_config"],
            nu_ehe_config=connected_sim_params["nu_ehe_config"],
            seed=connected_sim_params["seed"],
            flux_factor=ff,
            flare_only=True,
            det_only=True,
        )

        sim.run(parallel=True, n_jobs=2)

    time.sleep(1)

    BlazarNuConnectedResults.merge_over_flux_factor(
        sub_file_names,
        connected_sim_params["flux_factors"],
        write_to=output_directory.join("test_connected_sim_parallel.h5"),
    )

    results = BlazarNuConnectedResults.load(
        [output_directory.join("test_connected_sim_parallel.h5")]
    )

    assert len(results.bllac["n_alerts"]) == len(results.fsrq["n_alerts"]) == 2
