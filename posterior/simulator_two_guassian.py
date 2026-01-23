import json
from os import path
import numpy as np
import matplotlib.pyplot as plt

from autoconf.dictable import to_dict
import autofit as af


def simulate_dataset_1d_via_two_gaussians_from(
        gaussian_0,
        gaussian_1,
        dataset_path,
):
    """
    Simulate a 1D dataset composed of the sum of two Gaussian profiles.
    """

    pixels = 100
    xvalues = np.arange(pixels)

    """
    Evaluate both Gaussian components and sum them.
    """
    model_data_0 = gaussian_0.model_data_from(xvalues=xvalues)
    model_data_1 = gaussian_1.model_data_from(xvalues=xvalues)
    model_data_1d = model_data_0 + model_data_1

    """
    Add Gaussian noise.
    """
    signal_to_noise_ratio = 25.0
    noise = np.random.normal(0.0, 1.0 / signal_to_noise_ratio, pixels)

    data = model_data_1d + noise
    noise_map = (1.0 / signal_to_noise_ratio) * np.ones(pixels)

    """
    Write dataset to disk.
    """
    af.util.numpy_array_to_json(
        array=data,
        file_path=path.join(dataset_path, "data.json"),
        overwrite=True,
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

    plt.plot(xvalues, model_data_0, "--", label="Gaussian 0")
    plt.plot(xvalues, model_data_1, "--", label="Gaussian 1")
    plt.plot(xvalues, model_data_1d, "-", label="Total model")

    plt.legend()
    plt.title("1D Two-Gaussian Dataset")
    plt.xlabel("x values of profile")
    plt.ylabel("Profile normalization")
    plt.savefig(path.join(dataset_path, "image.png"))
    plt.close()

    """
    Persist both Gaussian components.
    """
    model_file = path.join(dataset_path, "model.json")
    with open(model_file, "w+") as f:
        json.dump(
            {
                "gaussian_0": to_dict(gaussian_0),
                "gaussian_1": to_dict(gaussian_1),
            },
            f,
            indent=4,
        )


total_datasets = 50

for i in range(total_datasets):
    dataset_path = path.join("dataset_dual", f"dataset_{i}")

    centre_prior = af.UniformPrior(
        lower_limit=20.0,
        upper_limit=80.0,
    )
    normalization_prior = af.UniformPrior(
        lower_limit=1.0,
        upper_limit=1e3,
    )
    sigma_prior = af.UniformPrior(
        lower_limit=1.0,
        upper_limit=100.0,
    )

    gaussian_0 = af.ex.Gaussian(
        centre=centre_prior.value_for(unit=float(np.random.random(1))),
        normalization=normalization_prior.value_for(unit=float(np.random.random(1))),
        sigma=sigma_prior.value_for(unit=float(np.random.random(1))),
    )

    gaussian_1 = af.ex.Gaussian(
        centre=centre_prior.value_for(unit=float(np.random.random(1))),
        normalization=normalization_prior.value_for(unit=float(np.random.random(1))),
        sigma=sigma_prior.value_for(unit=float(np.random.random(1))),
    )

    simulate_dataset_1d_via_two_gaussians_from(
        gaussian_0=gaussian_0,
        gaussian_1=gaussian_1,
        dataset_path=dataset_path,
    )
