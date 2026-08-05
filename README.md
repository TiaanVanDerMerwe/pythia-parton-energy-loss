# PYTHIA Parton Energy Loss

A Monte Carlo framework for exploring parton energy loss and its effects
on hadronic observables using **PYTHIA 8.317**.

This project investigates a simplified approach to modelling
medium-modified particle production by extracting partonic information
from PYTHIA events after the parton shower, applying an energy-loss
prescription to the outgoing partons, and studying how these
modifications propagate to final-state hadrons.

The framework also provides tools for calculating medium-modified
dihadron observables, including (I\_{AA}).

## Features

-   Proton--proton event generation with **PYTHIA 8.317**
-   Extraction of parton-level information from the PYTHIA event record
-   Identification of post-shower partons prior to hadronization
-   Storage and analysis of parton kinematic information
-   Application of simplified parton energy-loss prescriptions
-   Hadronization of modified partonic systems
-   Dihadron correlation analysis of the resulting hadrons
-   Calculation of (I\_{AA})
-   Comparison between vacuum and energy-loss scenarios
-   Plotting and visualization tools

## Requirements

-   C++17
-   PYTHIA 8.317
-   Python 3
-   NumPy
-   pandas
-   Matplotlib

## Disclaimer

The energy-loss implementation in this repository is intended for
methodological development and exploratory studies. It should not be
interpreted as a complete or precision description of parton propagation
through the quark--gluon plasma.

## Author

Tiaan van der Merwe

MSc Physics\
University of Cape Town
