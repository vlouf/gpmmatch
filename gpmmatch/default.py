"""
Volume matching of ground radar and GPM satellite. Default naming and attribute.

@title: gpmmatch
@author: Valentin Louf <valentin.louf@bom.gov.au>
@institutions: Monash University and the Australian Bureau of Meteorology
@creation: 24/02/2020
@date: 29/05/2025

.. autosummary::
    :toctree: generated/

    gpmset_metadata
    get_metadata
"""

from typing import Dict, Any


def gpmset_metadata() -> Dict[str, Any]:
    """
    Return a bunch of metadata (description, units, long_name, etc.) for the
    GPM set metadata.
    This metadata is used to describe the GPM data columns that are
    volume-matched to the ground radar data.
    The metadata includes information about the GPM overpass time, parallax
    corrected coordinates, precipitation inside the ground radar scope,
    range and elevation from the ground radar, and reflectivity in the
    ground radar band.

    Returns:
    ========
    metadata: dict
    """
    metadata = {
        "overpass_time": {"description": "GPM overpass time at the closest from ground radar site"},
        "x": {
            "units": "m",
            "description": "x-axis parallax corrected coordinates in relation to ground radar.",
        },
        "y": {
            "units": "m",
            "description": "y-axis parallax corrected coordinates in relation to ground radar.",
        },
        "z": {
            "units": "m",
            "description": "z-axis parallax corrected coordinates in relation to ground radar.",
        },
        "precip_in_gr_domain": {
            "units": "1",
            "description": "Satellite data-columns with precipitation inside the ground radar scope.",
        },
        "range_from_gr": {
            "units": "m",
            "description": "Range from satellite bins in relation to ground radar",
        },
        "elev_from_gr": {
            "units": "degrees",
            "description": "Elevation from satellite bins in relation to ground radar",
        },
        "reflectivity_grband": {"units": "dBZ"},
    }
    return metadata


def get_metadata() -> Dict[str, Dict[str, str]]:
    """
    Return a bunch of metadata (description, units, long_name, etc.) for the
    output dataset.

    Returns:
    ========
    metadata: dict
    """
    metadata = {
        "fmin_gpm": {
            "units": "",
            "long_name": "fmin_gpm",
            "description": "FMIN ratio for GPM.",
        },
        "fmin_gr": {
            "units": "",
            "long_name": "fmin_gr",
            "description": "FMIN ratio for GR.",
        },
        "refl_gpm_raw": {
            "units": "dBZ",
            "long_name": "GPM_reflectivity",
            "description": "GPM reflectivity volume-matched to ground radar.",
        },
        "refl_gpm_grband": {
            "units": "dBZ",
            "long_name": "GPM_reflectivity_grband_stratiform",
            "description": "GPM reflectivity converted to ground radar frequency band.",
        },
        "refl_gr_raw": {
            "units": "dBZ",
            "long_name": "reflectivity",
            "description": "Ground radar reflectivity volume matched using a `normal` average.",
        },
        "refl_gr_weigthed": {
            "units": "dBZ",
            "long_name": "reflectivity",
            "description": "Ground radar reflectivity volume matched using a distance-weighted average.",
        },
        "std_refl_gpm": {
            "units": "dB",
            "long_name": "standard_deviation_reflectivity",
            "description": "GPM reflectivity standard deviation of the volume-matched sample.",
        },
        "std_refl_gr": {
            "units": "dB",
            "long_name": "standard_deviation_reflectivity",
            "description": "Ground radar reflectivity standard deviation of the volume-matched sample.",
        },
        "sample_gpm": {
            "units": "1",
            "long_name": "sample_size",
            "description": "Number of GPM bins used to compute the volume-matched pixels at a given points",
        },
        "reject_gpm": {
            "units": "1",
            "long_name": "rejected_sample_size",
            "description": "Number of GPM bins rejected to compute the volume-matched pixels at a given points",
        },
        "sample_gr": {
            "units": "1",
            "long_name": "sample_size",
            "description": "Number of ground-radar bins used to compute the volume-matched pixels at a given points",
        },
        "reject_gr": {
            "units": "1",
            "long_name": "rejected_sample_size",
            "description": "Number of ground-radar bins rejected to compute the volume-matched pixels at a given points",
        },
        "volume_match_gpm": {
            "units": "m^3",
            "long_name": "volume",
            "description": "Volume of the GPM sample for each match points.",
        },
        "volume_match_gr": {
            "units": "m^3",
            "long_name": "volume",
            "description": "Volume of the ground radar sample for each match points.",
        },
        "x": {
            "units": "m",
            "long_name": "projected_x_axis_coordinates",
            "projection": "Azimuthal Equidistant from ground radar.",
        },
        "y": {
            "units": "m",
            "long_name": "projected_y_axis_coordinates",
            "projection": "Azimuthal Equidistant from ground radar.",
        },
        "z": {
            "units": "m",
            "long_name": "projected_z_axis_coordinates",
            "projection": "Azimuthal Equidistant from ground radar.",
        },
        "r": {
            "units": "m",
            "long_name": "range",
            "description": "Range from ground radar.",
        },
        "timedelta": {
            "units": "ns",
            "long_name": "timedelta",
            "description": "Maximum time delta between ground radar and GPM volumes.",
        },
        "elevation_gr": {
            "units": "degrees",
            "long_name": "elevation",
            "description": "Ground radar reference elevation.",
        },
        "ntilt": {
            "units": "1",
            "long_name": "ground_radar_tilt_number",
            "description": "Number of ground radar tilts used for volume matching.",
        },
        "nprof": {
            "units": "1",
            "long_name": "gpm_profile_number",
            "description": "Number of GPM profiles (nrays x nscan) used for volume matching.",
        },
        "pir_gpm": {
            "units": "dB m-1",
            "long_name": "GPM_path_integrated_reflectivity",
            "description": "Path integrated GPM reflectivity volume-matched.",
        },
        "pir_gr": {
            "units": "dB m-1",
            "long_name": "GR_path_integrated_reflectivity",
            "description": "Path integrated GR reflectivity volume-matched.",
        },
    }
    return metadata
