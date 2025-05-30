#!/usr/bin/env python
# -*- coding: utf-8 -*-
import io
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command

def parse_requirements(filename):
    with open(filename, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Package meta-data.
NAME = "gpmmatch"
DESCRIPTION = "Volume matching of ground radar and GPM satellite data."
URL = "https://github.com/vlouf/gpmmatch"
EMAIL = "valentin.louf@bom.gov.au"
AUTHOR = "Valentin Louf"

here = os.path.abspath(os.path.dirname(__file__))
with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = "\n" + f.read()

REQUIRED = parse_requirements("requirements.txt")
class PublishCommand(Command):
    """Support setup.py publish."""

    description = "Build and publish the package."
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print("\033[1m{0}\033[0m".format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status("Removing previous builds…")
            rmtree(os.path.join(here, "dist"))
        except FileNotFoundError:
            pass

        self.status("Building Source and Wheel (universal) distribution…")
        os.system("{0} setup.py sdist bdist_wheel --universal".format(sys.executable))

        self.status("Uploading the package to PyPi via Twine…")
        os.system("twine upload dist/*")

        sys.exit()


# Where the magic happens:
setup(
    name=NAME,
    version="1.3.0",
    description=DESCRIPTION,
    long_description=long_description,
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    project_urls={
        "Documentation": "https://github.com/vlouf/gpmmatch#readme",
        "Source": "https://github.com/vlouf/gpmmatch",
        "Tracker": "https://github.com/vlouf/gpmmatch/issues",
    },
    packages=find_packages(exclude=("tests", "example", "notebooks")),
    package_data={"gpmmatch": ["data/radar_site_list.csv"]},
    install_requires=REQUIRED,
    include_package_data=True,
    license="Apache-2.0",
    python_requires=">=3.9",
    classifiers=[
        # Trove classifiers
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Environment :: Jupyter",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
    ],
    # $ setup.py publish support.
    cmdclass={
        "publish": PublishCommand,
    },
)
