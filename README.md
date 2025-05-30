# 🛰️ gpmmatch 🛰️

gpmmatch is a Python library for matching satellite and ground-based radar volumes. It is primarily designed for use with GPM (Global Precipitation Measurement) data, but also supports the latest TRMM (Tropical Rainfall Measuring Mission) products

## Libraries needed:

- arm_pyart
- numpy
- pandas 
- netCDF4
- xarray
- dask
- pyodim

These libraries can be installed using pip:
```
pip install numpy pandas netCDF4 xarray dask arm_pyart
```

In addition, you will need to install the `gpmmatch` and `pyodim` libraries from Github:
```
pip install git+https://github.com/vlouf/gpmmatch.git
pip install pyodim`
```

## Example Jupyter Notebook

An example Jupyter notebook is available in the `example` directory. This notebook demonstrates how to use the `gpmmatch` library to a volume matching of GPM data against radar data. The notebook provides step-by-step instructions for downloading a sample of radar data from the Australian weather radar network archive. Finally, the notebook uses Matplotlib to create a plot of the results of the GPMmatch technique.

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
