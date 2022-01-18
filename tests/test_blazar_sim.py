from popsynth.population_synth import PopulationSynth

from nu_coincidence.utils.package_data import (
    get_available_config,
    get_path_to_config,
)

config_files = get_available_config()


def test_blazar_sim():

    blazar_config_files = [f for f in config_files if ("bllac" in f or "fsrq" in f)]

    for blazar_config_file in blazar_config_files:

        print(blazar_config_file)

        blazar_config = get_path_to_config(blazar_config_file)

        pop_gen = PopulationSynth.from_file(blazar_config)

        pop_gen._seed = 42

        pop = pop_gen.draw_survey()

        assert pop.distances.size > 0

        assert pop.distances[pop.selection].size > 0
