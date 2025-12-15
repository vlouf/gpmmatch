# 🛰️ gpmmatch 🛰️

gpmmatch is a Python library for matching satellite and ground-based radar volumes. It is primarily designed for use with GPM (Global Precipitation Measurement) data, but also supports the latest TRMM (Tropical Rainfall Measuring Mission) products.

## ⚠️ Branch and Radar Format Compatibility

> **Important Notice on Branch Usage and Radar Format Support**

The `master` branch of this repository uses **`pyodim`** for radar data I/O and is **only compatible with ODIM HDF5 radar format**. This branch is used in **operational environments**, and **ODIM is the only officially supported radar format** for production use.

If you are working with other radar formats (e.g., NEXRAD, UF, SIGMET, etc.), please use the `pyart` branch, which integrates with the **Py-ART** library and supports a wider range of radar data formats.

| Branch   | Radar I/O Library | Supported Formats         | Intended Use            |
|----------|-------------------|---------------------------|--------------------------|
| `master` | `pyodim`          | ODIM only                 | Operational (official)   |
| `pyart`  | `Py-ART`          | Multiple radar formats    | Research & development   |

Please ensure you are using the appropriate branch and radar I/O backend for your application.

---

## 🚀 Installation

Install gpmmatch directly from the repository:

```bash
pip install git+https://github.com/vlouf/gpmmatch.git
```

Or clone and install locally:

```bash
git clone https://github.com/vlouf/gpmmatch.git
cd gpmmatch
pip install .
```

## 🛠️ Dependencies

The following Python libraries are required to use `gpmmatch` (automatically installed with the package):

- `numpy` - Array operations and numerical computing
- `scipy` - Scientific computing and spatial operations
- `pandas` - Data manipulation and time handling
- `h5py` - HDF5 file reading for GPM data
- `netCDF4` - NetCDF file I/O
- `xarray` - Multi-dimensional labeled arrays and datasets
- `pyodim` - ODIM HDF5 radar format reader (master branch only)
- `pyproj` - Cartographic projections and coordinate transformations

**Python Version:** Requires Python 3.9 or later

## ✨ Key Features

- **Volume Matching**: Spatially and temporally match GPM satellite and ground radar observations
- **Calibration Monitoring**: Real-time monitoring of weather radar network calibration using GPM as reference
- **Phase-Aware DFR Conversion**: Advanced reflectivity conversion accounting for hydrometeor phase (ice, melting layer, liquid) using GPM bright band detection
- **Attenuation Correction**: Built-in attenuation correction for C-band and X-band radars using:
  - ZPHI method (preferred when KDP is available)
  - Gunn-East method (fallback)
- **Multi-Pass Processing**: Iterative volume matching with automatic offset computation and convergence detection
- **Parallax Correction**: Automatic geometric correction for satellite viewing angle

## 📓 Example Jupyter Notebook

A comprehensive example Jupyter notebook is available in the [notebooks](notebooks/) directory:

- **[notebooks/Example.ipynb](notebooks/Example.ipynb)**: Step-by-step tutorial demonstrating:
  - How to perform volume matching of GPM data against ground radar data
  - Downloading sample radar data from the Australian weather radar network archive
  - Processing and analyzing the matched data
  - Creating publication-quality plots using Matplotlib

## 📖 Usage

### Basic Volume Matching

```python
import gpmmatch

# Perform volume matching
matchset = gpmmatch.volume_matching(
    gpmfile="GPM_data.HDF5",
    grfile="radar_data.h5",
    gr_beamwidth=1.0,
    radar_band="C",
    refl_name="DBZH",
    fname_prefix="my_radar",
    phase_aware_dfr=True,  # Use phase-aware DFR conversion
    correct_attenuation=True  # Apply attenuation correction
)
```

### Multi-Pass Processing with Offset Convergence

```python
# Multi-pass processing with automatic offset computation
gpmmatch.vmatch_multi_pass(
    gpmfile="GPM_data.HDF5",
    grfile="radar_data.h5",
    radar_band="C",
    refl_name="DBZH",
    fname_prefix="my_radar",
    output_dir="./output",
    offset_thld=0.5,  # Convergence threshold in dB
    phase_aware_dfr=True
)
```

### Key Parameters

- `gpmfile`: Path to GPM HDF5 data file
- `grfile`: Path to ground radar ODIM HDF5 file
- `radar_band`: Radar frequency band ('S', 'C', or 'X')
- `refl_name`: Name of reflectivity field in radar data (e.g., 'DBZH', 'corrected_reflectivity')
- `gr_beamwidth`: Ground radar 3dB beamwidth in degrees
- `phase_aware_dfr`: Enable phase-aware DFR conversion (recommended, default: True)
- `correct_attenuation`: Apply attenuation correction for C/X-band (default: True)
- `kdp_name`: Name of KDP field for ZPHI attenuation correction (default: 'KDP')

## Citation

If you use `gpmmatch` in your research, please cite the following paper:

**Louf, Valentin, and Alain Protat**. *Real-Time Monitoring of Weather Radar Network Calibration and Antenna Pointing.* Journal of Atmospheric and Oceanic Technology, April 24, 2023. https://doi.org/10.1175/JTECH-D-22-0118.1.

BibTeX:
```bibtex
@article {Louf2023,
    author = "Valentin Louf and Alain Protat",
    title = "Real-Time Monitoring of Weather Radar Network Calibration and Antenna Pointing",
    journal = "Journal of Atmospheric and Oceanic Technology",
    year = "2023",
    publisher = "American Meteorological Society",    
    volume = "40",
    number = "7",
    doi = "10.1175/JTECH-D-22-0118.1",
    pages=  "823 - 844",    
}
```

Other references for the reflectivity conversion:

- Rain DFR: Louf, V., A. Protat, R. A. Warren, S. M. Collis, D. B. Wolff, S. Raunyiar, C. Jakob, and W. A. Petersen, 2019: An Integrated Approach to Weather Radar Calibration and Monitoring Using Ground Clutter and Satellite Comparisons. *J. Atmos. Oceanic Technol.*, **36**, 17–39, https://doi.org/10.1175/JTECH-D-18-0007.1

- Ice scattering: Tyynelä, J., J. Leinonen, A. Moisseev, and T. Nousiainen, 2011: Radar Backscattering from Snowflakes: Comparison of Fractal, Aggregate, and Soft Spheroid Models. *J. Atmos. Oceanic Technol.*, **28**, 1365–1372, https://doi.org/10.1175/JTECH-D-11-00004.1

- Ice scattering: Leinonen, J., 2014: High-level interface to T-matrix scattering calculations: architecture, capabilities and limitations. *Opt. Express*, **22**(2), 1655–1660, https://doi.org/10.1364/OE.22.001655

- GPM bright band: Awaka, J., M. Le, V. Chandrasekar, N. Yoshida, T. Higashiuwatoko, T. Kubota, and T. Iguchi, 2016: Rain Type Classification Algorithm Module for GPM Dual-Frequency Precipitation Radar. *J. Atmos. Oceanic Technol.*, **33**, 1887–1898, https://doi.org/10.1175/JTECH-D-16-0016.1

## License

This library is open source and made freely available according to the below
text:

    Copyright 2020 Valentin Louf
    Copyright 2023 Commonwealth of Australia, Bureau of Meteorology

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

A copy of the license is also provided in the LICENSE file included with the
source distribution of the library.
