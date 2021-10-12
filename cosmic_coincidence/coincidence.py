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

    All angles should be in radians.
    """

    n_match_spatial = 0
    spatial_match_inds = []

    # For each event
    for e_ra, e_dec, e_ang_err in zip(event_ras, event_decs, event_ang_errs):

        # Check if source locations inside event circle
        sigmas = get_central_angle(e_ra, e_dec, population_ras, population_decs)
        match_selection = sigmas <= e_ang_err

        n_match_spatial += len(match_selection[match_selection == True])

        # Indices of sources which match this event
        spatial_match_inds.append(np.where(match_selection == True)[0])

    return n_match_spatial, spatial_match_inds


def get_central_angle(ref_ra, ref_dec, ras, decs):
    """
    Get the central angles between ref_ra and
    ref_dec and a bunch of ras and decs. Angles
    should be in radians.

    Useful for computing the separation of points
    on the unit sphere.
    """

    sin_term = np.sin(ref_dec) * np.sin(decs)

    cos_term = np.cos(ref_dec) * np.cos(decs)

    diff_term = np.cos(ref_ra - ras)

    return np.arccos(sin_term + cos_term * diff_term)


def check_temporal_coincidence(
    event_times,
    spatial_match_inds,
    population_variability,
    population_flare_times,
    population_flare_durations,
    population_flare_amplitudes,
):
    """
    Check the temporal coincidence of events,
    both in the sense of variable objects and
    actual flares. Also requires a spatial match
    """

    n_match_variable = 0
    n_match_flaring = 0
    matched_flare_amplitudes = []

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

                    flare_amplitudes = population_flare_amplitudes[ind]

                    selection = (e_time >= flare_times) & (
                        e_time <= flare_times + flare_durations
                    )

                    matches = len(np.where(selection == True)[0])

                    # matches can *very rarely* be >1 for overlapping flares
                    # overlapping flares can occur if diff(flare_times) < 1 week
                    # in this case merge flares and take amplitude of first flare
                    if matches > 0:

                        n_match_flaring += 1

                        matched_flare_amplitudes.append(flare_amplitudes[selection][0])

    return n_match_variable, n_match_flaring, matched_flare_amplitudes
