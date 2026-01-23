"""
__Simulators__

These scripts simulates many 1D Gaussian datasets which are used to produce posteriors via model fitting
to train the aggregator agent.
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import json
from os import path
import numpy as np
import matplotlib.pyplot as plt

from autoconf.dictable import to_dict
import autofit as af


def simulate_dataset_1d_via_gaussian_from(gaussian, dataset_path):
    """
    Specify the number of pixels used to create the xvalues on which the 1D line of the profile is generated using and
    thus defining the number of data-points in our data.
    """
    pixels = 100
    xvalues = np.arange(pixels)

    """
    Evaluate this `Gaussian` model instance at every xvalues to create its model profile.
    """
    model_data_1d = gaussian.model_data_from(xvalues=xvalues)

    """
    Determine the noise (at a specified signal to noise level) in every pixel of our model profile.
    """
    signal_to_noise_ratio = 25.0
    noise = np.random.normal(0.0, 1.0 / signal_to_noise_ratio, pixels)

    """
    Add this noise to the model line to create the line data that is fitted, using the signal-to-noise ratio to compute
    noise-map of our data which is required when evaluating the chi-squared value of the likelihood.
    """
    data = model_data_1d + noise
    noise_map = (1.0 / signal_to_noise_ratio) * np.ones(pixels)

    """
    Output the data and noise-map to the `autofit_workspace/dataset` folder so they can be loaded and used 
    in other example scripts.
    """
    af.util.numpy_array_to_json(
        array=data, file_path=path.join(dataset_path, "data.json"), overwrite=True
    )
    af.util.numpy_array_to_json(
        array=noise_map,
        file_path=path.join(dataset_path, "noise_map.json"),
        overwrite=True,
    )
    plt.errorbar(
        x=xvalues,
        y=data,
        yerr=noise_map,
        linestyle="",
        color="k",
        ecolor="k",
        elinewidth=1,
        capsize=2,
    )
    plt.title("1D Gaussian Dataset.")
    plt.xlabel("x values of profile")
    plt.ylabel("Profile normalization")
    plt.savefig(path.join(dataset_path, "image.png"))
    plt.close()

    """
    __Model Json__

    Output the model to a .json file so we can refer to its parameters in the future.
    """
    model_file = path.join(dataset_path, "model.json")

    with open(model_file, "w+") as f:
        try:
            json.dump(to_dict(gaussian), f, indent=4)
        except (TypeError, ValueError):
            pass


total_datasets = 50

for i in range(total_datasets):
    dataset_path = path.join("dataset", f"dataset_{i}")

    centre_prior = af.UniformPrior(
        lower_limit=40.0,
        upper_limit=60.0,
    )
    normalization_prior = af.UniformPrior(
        lower_limit=1.0,
        upper_limit=1e2,
    )
    sigma_prior = af.UniformPrior(
        lower_limit=1.0,
        upper_limit=10.0,
    )

    centre = centre_prior.value_for(unit=float(np.random.random(1)))
    normalization_value = normalization_prior.value_for(unit=float(np.random.random(1)))
    sigma_value = sigma_prior.value_for(unit=float(np.random.random(1)))

    gaussian = af.ex.Gaussian(
        centre=centre, normalization=normalization_value, sigma=sigma_value
    )
    simulate_dataset_1d_via_gaussian_from(gaussian=gaussian, dataset_path=dataset_path)
