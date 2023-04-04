# 🛰️ gpmmatch 🛰️

`gpmmatch` is a Python library for ground-radar vs satellite volume-matching. It is built to work with GPM, but also supports the latest version of TRMM products.

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
pip install git+https://github.com/vlouf/pyodim.git`
```

## Example Jupyter Notebook

An example Jupyter notebook is available in the `example` directory. This notebook demonstrates how to use the `gpmmatch` library to a volume matching of GPM data against radar data. The notebook provides step-by-step instructions for downloading a sample of radar data from the Australian weather radar network archive. Finally, the notebook uses Matplotlib to create a plot of the results of the GPMmatch technique.

## References

If you use `gpmmatch` for a scientific publication, please cite the following paper:

Louf, V., Protat, A., Warren, R. A., Collis, S. M., Wolff, D. B., Raunyiar, S., Jakob, C., & Petersen, W. A. (2019). An Integrated Approach to Weather Radar Calibration and Monitoring Using Ground Clutter and Satellite Comparisons. Journal of Atmospheric and Oceanic Technology, 36(1), 17–39. [10.1175/JTECH-D-18-0007.1](https://doi.org/10.1175/JTECH-D-18-0007.1)

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
