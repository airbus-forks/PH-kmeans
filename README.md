# $k$-means Clustering in Persistent Homology

$k$-means clustering for use on persistent homology representations.

# Usage

An example notebook is available in the root folder here
[`./PHKmeans.ipynb`](./PHKmeans.ipynb)

# Contribute
The repository is structed as follows:
* `PD_subsample` contains code from [PD-subsample](https://anonymous.4open.science/r/PD-subsample-2321/), namely the `ApproxPH.py` script that we use and adapt code from to construct persistence measures and calculate distances between.
* `src` contains all code written for this project:
  *  `data_utils` contains scripts generating the simulated data and processing the real-world data used in the project. This directory also contains methods used to clean and process persistent homology representations.
  * `pd_pm_kmeans.py` contains the persistence diagram and persistence measure $k$-means algorithms.
  * `synthetic_data_clustering.ipynb` demonstrates how the code was used to generate results on simulated data.

# Project setup

Use the proect / package manager of your choice to install project dependancies.

Here is an example with `uv`
```
uv sync
```

# note

  * at first you must change project file name from PH-kmeans  to PHkmeans(without dash)
