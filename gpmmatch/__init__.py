"""
GPM volume matching package.

Volume matching of ground radar and GPM satellite data.
"""

from .gpmmatch import (
    NoRainError,
    generate_filename,
    get_radar_coordinates,
    get_gr_reflectivity,
    has_valid_data,
    volume_matching,
    vmatch_multi_pass,
    MIN_SAMPLE_POINTS,
    MIN_REFL_SAMPLES,
    MAX_OFFSET_THRESHOLD,
)

__author__ = "Valentin Louf"
__email__ = "valentin.louf@bom.gov.au"

__all__ = [
    "NoRainError",
    "generate_filename",
    "get_radar_coordinates",
    "get_gr_reflectivity",
    "has_valid_data",
    "volume_matching",
    "vmatch_multi_pass",
    "MIN_SAMPLE_POINTS",
    "MIN_REFL_SAMPLES",
    "MAX_OFFSET_THRESHOLD",
]
