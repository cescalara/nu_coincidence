from abc import ABCMeta
import numpy as np


class CoincidenceCheck(object, metaclass=ABCMeta):
    """
    Abstract base class for coincidence checks.
    """

    def __init__(self):

        pass

    def check_spatial(self):

        pass

    def check_temporal(self):

        pass


def check_spatial_coincidence(
    event_ras,
    event_decs,
    event_ang_errs,
    population_ras,
    population_decs,
):
    """
    Check the spatial coincidence of events
    assuming circular error regions with the
    sources in population, which are assumed to be points.
    """

    n_match_spatial = 0
    spatial_match_inds = []

    # For each event
    for e_ra, e_dec, e_ang_err in zip(event_ras, event_decs, event_ang_errs):

        # Check if source locations inside event circle
        match_selection = (population_ras - e_ra) ** 2 + (
            population_decs - e_decs
        ) ** 2 <= e_ang_err ** 2

        n_match_spatial += len(match_selection[match_selection == True])

        # Indices of sources which match this event
        spatial_match_inds.append(np.where(match_selection == True)[0])

    return n_match_spatial, spatial_match_inds


def check_temporal_coincidence(
    event_times,
    spatial_match_inds,
    population_variability,
    population_flare_times,
    population_flare_durations,
):
    """
    Check the temporal coincidence of events,
    both in the sense of variable objects and
    actual flares. Also requires a spatial match
    """

    n_match_variable = 0
    n_match_flaring = 0

    # For each event
    for e_time, match_inds in zip(event_times, spatial_match_inds):

        # If there are spatial matches
        if match_inds.size:

            # For each matched source
            for ind in match_inds:

                # If source is variable
                if population_variability[ind]:

                    n_match_variable += 1

                    flare_times = population_flare_times[ind]

                    flare_durations = population_flare_durations[ind]

                    selection = (e_time >= flare_times) & (
                        e_time <= flare_times + flare_durations
                    )

                    n_match_flaring += len(np.where(selection == True)[0])

    return n_match_variable, n_match_flaring
