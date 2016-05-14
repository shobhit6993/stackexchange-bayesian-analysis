Bayesian Analysis of Stackexchange data
=====================

Hierarchical Bayesian analysis is applied to model the response time for questions on Stackexchange websites.

### Dependencies
- *p7zip* used in `extract.sh` to extract .7z files in Stackexchange dump. Any other package can be used as long as the `7za` command in `extract.sh` is appropriately modified.
- [Stan](http://mc-stan.org/) for MCMC inference.
- R interface for Stan: [rstan](http://mc-stan.org/interfaces/rstan.html)
- Python interface for Stan: [PyStan](http://mc-stan.org/interfaces/pystan.html)

### Extracting compressed Stackexchange dump
Run `extract.sh` in the project directory to extract stackexchange dump. Individual .7z files are assumed to be located in `stackexchange` folder in the project directory, and the extracted files are stored in `extracted` folder. The `unpacked` directory, including the directory structure are automatically created by the script. Additionally, it copies the directory structure of `unpacked` folder to `data` folder, for use by `parser.py`. The directory names can be changed in `extract.sh`

### Generating csv files
Run `parser.py` to generate csv files containing relevant attributes. The generated csv files are stored in `data` directory. The directory name can be changed in `extract.sh` and `parser.py`

### Models used
- Ordinary Least Squares Regression: Implemented in `linear-regression.R`
- Bayesian Linear Regression: Implemented in `hierarchical-linear-regression.R`, with the model specification in `bayesian-regression-pooled.stan`
- Hierarchical Linear Regression: Implemented in `hierarchical-linear-regression.R`, with the model specification in `bayesian-regression-hierarchical.stan`
- Generalized Linear Model: Using Exponential and Gamma link functions. Implemented in `stan_model.py`
- Probit Regression: Discretizing the regression problem. Implemented in `stan_model.py`
