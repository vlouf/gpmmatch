'''
Volume matching of ground radar and GPM satellite. Default naming and attribute.

@title: gpmmatch
@author: Valentin Louf <valentin.louf@bom.gov.au>
@institutions: Monash University and the Australian Bureau of Meteorology
@creation: 24/02/2020
@date: 05/05/2020
    get_metadata
'''


def get_metadata():
    '''
    Return a bunch of metadata (description, units, long_name, etc.) for the
    output dataset.

    Returns:
    ========
    metadata: dict
    '''
    metadata = {
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
        "zrefl_gpm_raw": {
            "units": "dBZ",
            "long_name": "GPM_reflectivity",
            "description": "GPM reflectivity volume-matched to ground radar computed in linear units.",
        },
        "zrefl_gpm_strat": {
            "units": "dBZ",
            "long_name": "GPM_reflectivity_grband_stratiform",
            "description": "GR-frequency band converted GPM reflectivity volume-matched to ground radar - stratiform approximation computed in linear units.",
        },
        "zrefl_gpm_conv": {
            "units": "dBZ",
            "long_name": "GPM_reflectivity_grband_convective",
            "description": "GR-frequency band converted GPM reflectivity volume-matched to ground radar - convective approximation computed in linear units.",
        },
        "zrefl_gr_raw": {
            "units": "dBZ",
            "long_name": "reflectivity",
            "description": "Ground radar reflectivity volume matched using a `normal` average computed in linear units.",
        },
        "zrefl_gr_weigthed": {
            "units": "dBZ",
            "long_name": "reflectivity",
            "description": "Ground radar reflectivity volume matched using a distance-weighted average computed in linear units.",
        },
        "std_zrefl_gpm": {
            "units": "dB",
            "long_name": "standard_deviation_reflectivity",
            "description": "GPM reflectivity standard deviation of the volume-matched sample computed in linear units.",
        },
        "std_zrefl_gr": {
            "units": "dB",
            "long_name": "standard_deviation_reflectivity",
            "description": "Ground radar reflectivity standard deviation of the volume-matched sample computed in linear units.",
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