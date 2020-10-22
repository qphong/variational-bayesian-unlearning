
# Variational Bayesian Unlearning

### Prerequisites

```
python = 3.7
tensorflow = 1.14.0
tensorflow-probability = 0.7.0
matplotlib
numpy
scipy
pickle
argparse
```

### Running the experiments

## Synthetic Moon Classification Dataset

1. To run the training with VI on full data, retraining with VI on remaining data, and unlearning using EUBO, rKL
```
python run.py --exper moon --appr gauss_fullcov --nsample 1000 --ntrain 30000 --folder result
```
The result is written to folder `result`.

2. To compute the averaged KL distance between approximate predictive distributions
```
python compute_kl_distance.py --folder result --outfolder plot_data --exper moon --appr gauss_fullcov
```
The KL distances are written to folder `plot_data`.

3. To plot the averaged KL distance between approximate predictive distributions
```
python plot_kl_distance_mean_std.py --folder plot_data --exper moon --appr gauss_fullcov
```

4. To plot the posterior mean and variance of the latent function
```
python plot_moon_gauss.py --exper moon
```

5. To plot the predictive distributions
```
python plot_moon_marginal_prob.py --exper moon
```

By changing `moon` in `--expr moon` with `moon_random`, `moon_rm_30`, `moon_rm_40`, `moon_rm_50`, we can generate the result for the 4 scenarios in Appendix D.


## Banknote Authentication Dataset

1. To run the training with VI on full data, retraining with VI on remaining data, and unlearning using EUBO, rKL where the approximate posterior beliefs are modeled by
    * Multivariate Gaussians with full covariance matrices
    ```
    python run.py --exper banknote_authentication1 --appr gauss_fullcov --nsample 1000 --ntrain 30000 --folder result
    ```

    * Normalizing flows
    ```
    python run.py --exper banknote_authentication1 --appr maf --nbijector 15 --nhidden 5 --nsample 1000 --ntrain 30000 --folder result
    ```
The result is written to folder `result`.
In the following step, we can change `maf` in `--appr maf` to `gauss_fullcov`

2. To compute the averaged KL distance between approximate posterior beliefs
```
python compute_kl_distance.py --folder result --outfolder plot_data --exper banknote_authentication1 --appr maf
```
The KL distances are written to folder `plot_data`.

3. To plot the averaged KL distance between approximate predictive distributions
```
python plot_kl_distance_mean_std.py --folder plot_data --exper banknote_authentication1 --appr maf
```
