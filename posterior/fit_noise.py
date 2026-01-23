"""
Overview: The Basics
--------------------

**PyAutoFit** is a Python based probabilistic programming language for model fitting and Bayesian inference
of large datasets.

The basic **PyAutoFit** API allows us a user to quickly compose a probabilistic model and fit it to data via a
log likelihood function, using a range of non-linear search algorithms (e.g. MCMC, nested sampling).

This overview gives a run through of:

 - **Models**: Use Python classes to compose the model which is fitted to data.
 - **Instances**: Create instances of the model via its Python class.
 - **Analysis**: Define an ``Analysis`` class which includes the log likelihood function that fits the model to the data.
 - **Searches**: Choose an MCMC, nested sampling or maximum likelihood estimator non-linear search algorithm that fits the model to the data.
 - **Model Fit**: Fit the model to the data using the chosen non-linear search, with on-the-fly results and visualization.
 - **Results**: Use the results of the search to interpret and visualize the model fit.
 - **Samples**: Use the samples of the search to inspect the parameter samples and visualize the probability density function of the results.
 - **Multiple Datasets**: Dedicated support for simultaneously fitting multiple datasets, enabling scalable analysis of large datasets.

This overviews provides a high level of the basic API, with more advanced functionality described in the following
overviews and the **PyAutoFit** cookbooks.

To begin, lets import ``autofit`` (and ``numpy``) using the convention below:
"""
from random import random, uniform

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import autofit as af
import numpy as np

from os import path


class Analysis(af.Analysis):
    def log_likelihood_function(self, instance: af.ModelInstance, xp=np) -> float:
        """
        Determine the log likelihood of a fit of multiple profiles to the dataset.

        Parameters
        ----------
        instance : af.Collection
            The model instances of the profiles.

        Returns
        -------
        The log likelihood value indicating how well this model fit the dataset.
        """
        return uniform(-3, 3)


total_datasets = 50

for i in range(total_datasets):
    dataset_path = path.join("dataset", f"dataset_{i}")

    data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
    noise_map = af.util.numpy_array_from_json(
        file_path=path.join(dataset_path, "noise_map.json")
    )

    model = af.Model(af.ex.Gaussian)
    print("Model `Gaussian` object: \n")
    print(model)

    model.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
    model.normalization = af.UniformPrior(lower_limit=0.0, upper_limit=1e2)
    model.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=30.0)

    analysis = Analysis(data=data, noise_map=noise_map)

    search = af.DynestyStatic(
        nlive=50,  # Example how to customize the search settings
        path_prefix="fit_noise",
        name=f"dataset_{i}_fit",
    )

    print(
        """
        The non-linear search has begun running.
        This Jupyter notebook cell with progress once the search has completed - this could take a few minutes!
        """
    )

    result = search.fit(model=model, analysis=analysis)
