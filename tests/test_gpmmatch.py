"""
Unit tests for gpmmatch module.

Tests volume matching of ground radar and GPM satellite data.
"""

import shutil
import pytest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

# Import from the specific module file, not the package
from gpmmatch.gpmmatch import (
    generate_filename,
    get_radar_coordinates,
    get_gr_reflectivity,
    has_valid_data,
    volume_matching,
    vmatch_multi_pass,
)


@pytest.fixture
def sample_radar_dataset():
    """Create a minimal radar dataset for testing."""
    ntime = 10
    nrange = 50

    radar = xr.Dataset(
        {
            "reflectivity": (["time", "range"], np.random.uniform(10, 50, (ntime, nrange))),
            "corrected_reflectivity": (["time", "range"], np.random.uniform(10, 50, (ntime, nrange))),
        },
        coords={
            "time": pd.date_range("2018-11-20 05:36:30", periods=ntime, freq="10s"),
            "range": np.arange(1000, 1000 + nrange * 250, 250),
            "x": (["time", "range"], np.random.uniform(-50000, 50000, (ntime, nrange))),
            "y": (["time", "range"], np.random.uniform(-50000, 50000, (ntime, nrange))),
            "azimuth": (["time"], np.linspace(0, 360, ntime)),
        },
        attrs={
            "start_time": "2018-11-20T05:36:30Z",
            "end_time": "2018-11-20T05:38:00Z",
            "longitude": 145.0,
            "latitude": -37.8,
        }
    )

    radar["elevation"] = (["time"], np.ones(ntime) * 0.5)

    return radar


@pytest.fixture
def sample_gpm_dataset():
    """Create a minimal GPM dataset for testing."""
    nscan = 20
    nray = 49
    nbin = 176

    gpm = xr.Dataset(
        {
            "zFactorCorrected": (["nscan", "nray", "nbin"],
                                np.random.uniform(10, 45, (nscan, nray, nbin))),
            "reflectivity_grband": (["nscan", "nray", "nbin"],
                                   np.random.uniform(10, 45, (nscan, nray, nbin))),
            "precip_in_gr_domain": (["nscan", "nray"],
                                   np.random.choice([0, 1], size=(nscan, nray), p=[0.3, 0.7])),
            "elev_from_gr": (["nscan", "nray", "nbin"],
                           np.random.uniform(0.5, 2.5, (nscan, nray, nbin))),
            "flagPrecip": (["nscan", "nray"], np.random.choice([0, 1, 2], size=(nscan, nray))),
        },
        coords={
            "nscan": pd.date_range("2018-11-20 05:33:41", periods=nscan, freq="1s"),
            "nray": np.arange(nray),
            "x": (["nscan", "nray", "nbin"], np.random.uniform(-100000, 100000, (nscan, nray, nbin))),
            "y": (["nscan", "nray", "nbin"], np.random.uniform(-100000, 100000, (nscan, nray, nbin))),
            "z": (["nscan", "nray", "nbin"], np.random.uniform(1000, 20000, (nscan, nray, nbin))),
        },
        attrs={
            "orbit": 26860,
        }
    )

    gpm["dr"] = 125.0
    gpm["beamwidth"] = 0.71
    gpm["altitude"] = 407000.0
    gpm["earth_gaussian_radius"] = 6371000.0
    gpm["distance_from_sr"] = (["nbin"], np.arange(nbin) * 125.0)
    gpm["overpass_time"] = pd.Timestamp("2018-11-20 05:37:00")

    return gpm


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for outputs."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


# Test helper functions
class TestHelperFunctions:

    def test_generate_filename(self, sample_radar_dataset, sample_gpm_dataset):
        """Test filename generation."""
        filename = generate_filename(sample_radar_dataset, sample_gpm_dataset, "test_radar")

        assert "vmatch.gpm.orbit" in filename
        assert "0026860" in filename
        assert "test_radar" in filename
        assert ".nc" in filename
        assert "20181120" in filename

    def test_get_radar_coordinates_no_offset(self, sample_radar_dataset):
        """Test radar coordinate extraction without elevation offset."""
        nradar = [sample_radar_dataset]

        range_gr, elev_gr, xradar, yradar, time_radar = get_radar_coordinates(nradar)

        assert len(range_gr) == 50
        assert len(elev_gr) == 1
        assert elev_gr[0] == pytest.approx(0.5)
        assert len(xradar) == 1
        assert len(yradar) == 1
        assert len(time_radar) == 1

    def test_get_radar_coordinates_with_offset(self, sample_radar_dataset):
        """Test radar coordinate extraction with elevation offset."""
        nradar = [sample_radar_dataset]
        elevation_offset = 0.3

        range_gr, elev_gr, xradar, yradar, time_radar = get_radar_coordinates(
            nradar, elevation_offset=elevation_offset
        )

        assert elev_gr[0] == pytest.approx(0.8)

    def test_get_gr_reflectivity(self, sample_radar_dataset):
        """Test ground radar reflectivity extraction."""
        nradar = [sample_radar_dataset]

        refl, pir = get_gr_reflectivity(nradar, "corrected_reflectivity", gr_offset=0, gr_refl_threshold=10)

        assert len(refl) == 1
        assert len(pir) == 1
        assert isinstance(refl[0], np.ma.MaskedArray)
        assert refl[0].shape == (10, 50)

    def test_get_gr_reflectivity_with_offset(self, sample_radar_dataset):
        """Test reflectivity extraction with offset applied."""
        nradar = [sample_radar_dataset]
        gr_offset = 5.0

        refl, pir = get_gr_reflectivity(
            nradar, "corrected_reflectivity", gr_offset=gr_offset, gr_refl_threshold=10
        )

        # Check that non-masked values are reduced by offset (accounting for threshold masking)
        original = sample_radar_dataset["corrected_reflectivity"].values
        valid_mask = ~refl[0].mask
        assert np.all(refl[0].data[valid_mask] <= original[valid_mask])

    def test_has_valid_data_valid(self):
        """Test has_valid_data with valid array."""
        arr = np.ma.masked_array([10.0, 20.0, 30.0, 40.0, 50.0], mask=[False, False, False, False, False])
        assert has_valid_data(arr, min_valid=5) is True

    def test_has_valid_data_insufficient(self):
        """Test has_valid_data with insufficient samples."""
        arr = np.ma.masked_array([10.0, 20.0, 30.0], mask=[False, False, False])
        assert has_valid_data(arr, min_valid=5) is False

    def test_has_valid_data_all_nan(self):
        """Test has_valid_data with all NaN values."""
        arr = np.ma.masked_array([np.nan, np.nan, np.nan, np.nan, np.nan],
                                 mask=[True, True, True, True, True])
        assert has_valid_data(arr, min_valid=5) is False


# Integration tests
class TestVolumeMatchingIntegration:

    @pytest.fixture
    def test_data_dir(self):
        """Return path to test data directory."""
        # Adjust this path to where your test files are located
        return Path("tests/data")

    @pytest.fixture
    def radar_file(self, test_data_dir):
        """Path to test radar file."""
        return test_data_dir / "02_20181120_053630.pvol.h5"

    @pytest.fixture
    def gpm_file(self, test_data_dir):
        """Path to test GPM file."""
        return test_data_dir / "2A-CS-AUS-East.GPM.Ku.V8-20180723.20181120-S053341-E054123.026860.V06A.HDF5"

    @pytest.mark.skipif(not Path("tests/data").exists(), reason="Test data not available")
    def test_volume_matching_real_data(self, radar_file, gpm_file, temp_output_dir):
        """Test volume matching with real data files."""
        if not radar_file.exists() or not gpm_file.exists():
            pytest.skip("Test data files not found")

        # Use DBZH instead of corrected_reflectivity based on actual file structure
        result = volume_matching(
            gpmfile=str(gpm_file),
            grfile=str(radar_file),
            gr_offset=0,
            gr_beamwidth=1.0,
            radar_band="C",
            refl_name="DBZH",
            fname_prefix="test_radar"
        )

        assert isinstance(result, xr.Dataset)
        assert "refl_gpm_raw" in result
        assert "refl_gr_raw" in result
        assert "offset_found" in result.attrs
        assert "final_offset" in result.attrs
        assert result.attrs["gpm_orbit"] == 26860

    @pytest.mark.skipif(not Path("tests/data").exists(), reason="Test data not available")
    def test_vmatch_multi_pass_real_data(self, radar_file, gpm_file, temp_output_dir):
        """Test multi-pass volume matching with real data."""
        if not radar_file.exists() or not gpm_file.exists():
            pytest.skip("Test data files not found")

        # Use DBZH instead of corrected_reflectivity based on actual file structure
        vmatch_multi_pass(
            gpmfile=str(gpm_file),
            grfile=str(radar_file),
            gr_offset=0,
            gr_beamwidth=1.0,
            gr_refl_threshold=10,
            radar_band="C",
            refl_name="DBZH",
            fname_prefix="test_radar",
            offset_thld=0.5,
            output_dir=str(temp_output_dir)
        )

        # Check that output files were created
        first_pass_dir = temp_output_dir / "first_pass"
        final_pass_dir = temp_output_dir / "final_pass"

        assert first_pass_dir.exists()
        assert final_pass_dir.exists()

        first_pass_files = list(first_pass_dir.glob("*.nc"))
        final_pass_files = list(final_pass_dir.glob("*.nc"))

        assert len(first_pass_files) > 0
        assert len(final_pass_files) > 0

        # Load and check final result
        final_file = final_pass_files[0]
        final_dataset = xr.open_dataset(final_file)

        assert "iteration_number" in final_dataset.attrs
        assert "offset_history" in final_dataset.attrs

        final_dataset.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])